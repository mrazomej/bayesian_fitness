# Import differential equations package
using DifferentialEquations

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import basic statistical functions
import StatsBase
import Distributions
import Random

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Logistic growth as data generating process
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    `dndt_rhs!(dx, x, param, t)``

Function to compute the right-hand side (rhs) of the logistic growth dynamics to
be used for the numerical integration routine.

# Arguments
- `dx::AbstractVector`: Pre-allocated output array containing the time
  derivative of all strains cell number.
- `x::AbstractVector`: Vector containing current cell copy number for all
  strains.
- `param::Tuple`: List of all parameters feeded into the function. The list of
  parameters must contain:
  - `λ::AbstractVector`: Array containing all of the growth rates for
    each of the strains.
  - `κ::AbstractFloat`: Media carrying capacity

- `t::AbstractFloat`: Time where to evaluate derivative. This is an input
  required for the numerical integration procedure.
"""
function dndt_rhs!(dx, x, param, t)
    # Unpack parameters
    λ, κ = param

    # return right-hand side of ODE
    @. dx = λ * x * (1 - ($(sum(x) / κ)))
end # function

@doc raw"""
    logistic_fitness_measurement(λ̲, n̲₀; n_cycles, n_gen, exp_frac, κ, dilution, poisson_noise, σ_lognormal)

Function to simulate fitness measurment experiments over multiple
growth-dilution cycles using a logistic growth model of the form

ṅᵢ = λᵢnᵢ (1 - ∑ᵢ(nᵢ)/κ),

and a Poisson sampling at each dilution step.

# Arguments
- `λ̲::Vector{Float64}`: Vector of growth rate values for all genotypes. NOTE:
  The reference growth rate (usually the neutral lineage) should be the *first
  entry* of this vector.
- `n̲₀::Vector{Int64}`: Vector with the initial frequencies for all genotypes to
  track.

# Optional Keyword Arguments
- `n_cycles::Int64=4`: Number of growth-dilution cycles over which to integrate
  the dynamics.
- `n_gen::Int64=8`: Number of generations that the reference strain goes through
  for every growth cycle.
- `exp_frac::Float64=1/3`: Fraction of the growth-dilution cycle that cells
  spend on exponential growth. NOTE: The total time of integration is computed
  as
  - `log(2) / first(λ̲) * n_gen / exp_frac`
- `κ::Float64=10.0^10`: Environment carrying capacity.
- `dilution::Union{Nothing,Float64}=nothing`: Dilution factor after every growth
  cycle. The initial condition for the next cycle are computed as
  - `solve(prob).u[end] ./ dilution`. if `typeof(dilution) == Nothing` the
  dilution factor is `1 / 2^n_gen`.
- `poisson_noise::Bool=true`: Boolean indicating if Poisson noise between
  dilutions should be added to the data.
- `σ_lognormal::Float64=0.0`: Standard deviation for a log-normal distribution
  that adds extra noise to th ebarcode measurements.

# Returns
- `Matrix{Int64}`: `(n_cycles + 1) × n_genotypes` matrix with the barcode count
  trajectories for all genotypes.
"""
function logistic_fitness_measurement(
    λ̲::Vector{Float64},
    n̲₀::Vector{Int64};
    n_cycles::Int64=4,
    n_gen::Int64=8,
    exp_frac::Float64=1 / 3,
    κ::Float64=10.0^10,
    dilution::Union{Nothing,Float64}=nothing,
    poisson_noise::Bool=true,
    σ_lognormal::Float64=0.0
)
    # Check the input exp_frac
    if !(0 < exp_frac < 1)
        error("exp_frac must be ∈ (0, 1]")
    end # if
    # Check the positivity of initial populations
    if any(n̲₀ .< 0)
        error("Initial populations cannot be negative")
    end # if
    # Check that the initial population is not larger than the carrying capacity
    if sum(n̲₀) ≥ κ
        error("The initial population is larger than the carrying capacity")
    end # if
    # Check the value of the σ_lognormal
    if σ_lognormal < 0.0
        error("σ_lognormal is strictly ≥ 0.0")
    end # if

    # Define time span
    t_span = (0.0, log(2) / first(λ̲) * (n_gen / exp_frac))

    # Define number of genotypes
    n_geno = length(λ̲)

    # Inititalize matrix to save output
    n_mat = Matrix{Int64}(undef, n_cycles + 1, n_geno)

    # Store initial condition
    n_mat[1, :] = n̲₀

    # Loop through cycles
    for cyc = 2:n_cycles+1
        # Initialize vector with inital conditions as zeros
        n_init = zeros(length(n̲₀))
        # Locate non-zero entries from previous cycle
        non_zero_idx = n_mat[cyc-1, :] .> 0.0

        if poisson_noise
            # Define initial condition for next cycle with Poisson sampling.
            n_init[non_zero_idx] = Random.rand.(
                Distributions.Poisson.(n_mat[cyc-1, non_zero_idx])
            )
        else
            n_init[non_zero_idx] = n_mat[cyc-1, non_zero_idx]
        end # if
        # Define ODE problem
        prob = ODEProblem(
            dndt_rhs!, n_init, t_span, (λ̲, κ)
        )
        # Check dilution factor
        if typeof(dilution) <: Nothing
            # Solve system and store final point
            n_mat[cyc, :] = Int64.(round.(solve(prob).u[end] ./ 2^(n_gen)))
        else
            # Solve system and store final point
            n_mat[cyc, :] = Int64.(round.(solve(prob).u[end] ./ dilution))
        end # if
    end # for

    # Add Gaussian noise to each value
    if σ_lognormal > 0.0
        # Sample Gaussian noise for each measurement
        n_mat[n_mat.>0] = Int64.(
            round.(
                exp.(
                    first.(
                        rand.(
                            Distributions.Normal.(
                                log.(n_mat[n_mat.>0]), Ref(σ_lognormal)
                            ),
                            1
                        )
                    )
                )
            )
        )
    end # if

    return n_mat
end # function