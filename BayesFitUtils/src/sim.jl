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
- `λ̲::AbstractVector{Float64}`: Vector of growth rate values for all genotypes.
  NOTE: The reference growth rate (usually the neutral lineage) should be the
  *first entry* of this vector.
- `n̲₀::AbstractVector{Int64}`: Vector with the initial frequencies for all
  genotypes to track.

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
    λ̲::AbstractVector{Float64},
    n̲₀::AbstractVector{Int64};
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

@doc raw"""
    logistic_fitness_measurement(λ̲̲, n̲₀; n_cycles, n_gen, exp_frac, κ, dilution, poisson_noise, σ_lognormal)

Function to simulate fitness measurment experiments over multiple
growth-dilution cycles using a logistic growth model of the form

ṅᵢ = λᵢnᵢ (1 - ∑ᵢ(nᵢ)/κ),

and a Poisson sampling at each dilution step. This method allows for different
growth rates on each of the environments.

# Arguments
- `λ̲̲::AbstractMatrix{Float64}`: `B × E` matrix of growth rate values for all
  `B` genotypes in all `E` unique environments. NOTE: The reference growth rate
  (usually the neutral lineage) should be the *first entry* of this vector.
- `n̲₀::AbstractVector{Int64}`: Vector with the initial frequencies for all
  genotypes to track.
- `env_idx::AbstractVector{Int64}`: Vector used to index the corresponding
  growth rate for each cycle. The number of entries in this vector defines the
  number of cycles.

# Optional Keyword Arguments
- `exp_frac::Float64=1/3`: Fraction of the growth-dilution cycle that cells
  spend on exponential growth. NOTE: The total time of integration is computed
  as
  - `log(2) / first(λ̲̲) * n_gen / exp_frac`. **Note** that the very first entry
  of the growth rate matrix is taken as the reference.
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
    λ̲̲::AbstractMatrix{Float64},
    n̲₀::AbstractVector{Int64},
    env_idx::Vector{Int64};
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
    t_span = (0.0, log(2) / first(λ̲̲) * (n_gen / exp_frac))

    # Define number of genotypes
    n_geno = size(λ̲̲, 1)

    # Define number of unique environments
    n_env = size(λ̲̲, 2)

    # Define number of cycles
    n_cycles = length(env_idx)

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
            dndt_rhs!, n_init, t_span, (λ̲̲[:, env_idx[cyc-1]], κ)
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Simulations as in Fit-Seq 2.0 with explicit PCR amplification
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

"""
    lineage_growth(cell_num::Real, fitness::Float64) -> Real

Calculate the growth of a single lineage over one time step using the Malthusian
fitness model.

# Arguments
- `cell_num::Real`: The number of cells in the lineage at the current time step.
- `fitness::Float64`: The Malthusian fitness of the lineage.

# Returns
`Real`: The calculated number of cells in the lineage at the next time step
before any sampling or noise is applied.

# Example
```julia
growth = lineage_growth(500, 0.05)
```
"""
function lineage_growth(
    cell_num::Real,
    fitness::Real
)
    # Compute lineage growth according to Malthusian fitness model
    return 2 * cell_num * exp(fitness)
end # function

## ----------------------------------------------------------------------------- 

"""
    growth_lineages(n_cells::Vector{<:Real}, fitness::Vector{<:Real}) -> Vector{Int}

Simulate the number of cells at time t+1 for multiple lineages, incorporating
stochastic effects via Poisson noise.

# Arguments
- `n_cells::Vector{<:Real}`: Vector containing the number of cells in each
  lineage at the current time step.
- `fitness::Vector{Float64}`: Vector containing the Malthusian fitness for each
  lineage.

# Returns
`Vector{Int}`: A vector containing the simulated number of cells for each
lineage at the next time step, after Poisson noise is applied.

# Example
```julia
new_cell_counts = growth_lineages([500, 300], [0.05, 0.02])
```
"""
function growth_lineages(
    n_cells::Vector{<:Real},
    fitness::Vector{<:Real}
)
    # Compute the number of cells at time t+1 given the number of cells at time
    # t
    n_cells_new = [lineage_growth(n, f) for (n, f) in zip(n_cells, fitness)]

    # Set all numbers less than one to zero
    n_cells_new[n_cells_new.<1.0] .= 0.0

    # Sample number of cells according to Poisson distribution
    return Random.rand.(Distributions.Poisson.(n_cells_new))
end # function

"""
    growth_lineages(n_cells::Vector{<:Real}, fitness::Vector{Float64}, 
                    n_gen::Int) -> Vector{Int}

Simulate the number of cells for multiple lineages over multiple generations,
incorporating stochastic effects via Poisson noise.

# Arguments
- `n_cells::Vector{<:Real}`: Vector containing the number of cells in each
  lineage at the initial time step.
- `fitness::Vector{Float64}`: Vector containing the Malthusian fitness for each
  lineage.
- `n_gen::Int`: The number of generations to simulate.

# Returns
`Vector{Int}`: A vector containing the simulated number of cells for each
lineage at the final generation after Poisson noise is applied.

# Description
This function simulates the growth of lineages over `n_gen` generations. It
first calculates the growth for the initial generation using the
`growth_lineages` function for one time step. It then iteratively applies this
function to simulate each subsequent generation, storing the results in a matrix
where each column represents a generation. The final population sizes for each
lineage after `n_gen` generations are returned.

# Example
```julia
final_cell_counts = growth_lineages([500, 300], [0.05, 0.02], 10)
```
"""
function growth_lineages(
    n_cells::Vector{<:Real},
    fitness::Vector{<:Real},
    n_gen::Int
)
    # Initialize array to save output
    n_cells_mat = Matrix{Int64}(undef, length(n_cells), n_gen)
    # Set initial condition
    n_cells_mat[:, 1] = growth_lineages(n_cells, fitness)

    if n_gen == 1
        return vec(n_cells_mat)
    else
        # Loop through generations
        for gen = 2:n_gen
            n_cells_mat[:, gen] = growth_lineages(
                n_cells_mat[:, gen-1], fitness
            )
        end # for

        # Return final piont
        return vec(n_cells_mat[:, end])
    end # if
end # function


## ----------------------------------------------------------------------------- 

"""
    transfer_lineages(n_cells::Vector{<:Real}, n_gen::Real) -> Vector{Int}

Simulate the number of cells to be transferred to the next growth batch after a
given number of generations.

# Arguments
- `n_cells::Vector{<:Real}`: Vector containing the number of cells in each
  lineage at the current time step.
- `n_gen::Real`: The number of generations that have passed.

# Returns
`Vector{Int}`: A vector containing the number of cells to be transferred for
each lineage, sampled using a Poisson distribution.

# Description
The function models the reduction of cell numbers due to the bottleneck effect
occurring during transfer to a new batch. The number of cells for each lineage
is assumed to be divided by 2^n_gen, and the actual number of transferred cells
is then sampled from a Poisson distribution centered around this expected value.

# Example
```julia
transferred_cells = transfer_lineage([1000, 1500], 3)
```
"""
function transfer_lineages(
    n_cells::Vector{<:Real},
    n_gen::Real,
)
    # Sample cells to be transferred to the next batch growth
    return Random.rand.(Distributions.Poisson.(n_cells ./ 2^n_gen))
end # function

## ----------------------------------------------------------------------------- 

"""
    sample_lineages(n_cells::Vector{<:Real}, L::Real=500) -> Vector{Int}

Simulate the sampling of cells from each lineage for PCR amplification based on
their relative abundance.

# Arguments
- `n_cells::Vector{<:Real}`: Vector containing the number of cells in each
  lineage.
- `L::Real`: The target total number of cells per barcode to sample for PCR.
  Defaults to 500 if not specified.

# Returns
`Vector{Int}`: A vector containing the sampled number of cells for each lineage,
determined by Poisson sampling based on the lineage's relative abundance.

# Description
The function calculates the fraction of the total population that each lineage
represents. It then uses these fractions to determine how many cells from each
lineage should be sampled to reach a total of `L` cells per lineage on average,
assuming that `L` cells are sampled from the entire population proportionally to
the abundance of each lineage. The actual numbers sampled are determined by a
Poisson distribution to introduce stochasticity into the sampling process.

# Example
```julia
sampled_cells = sample_lineages([100, 400, 500], 500)
```
"""
function sample_lineages(
    n_cells::Vector{<:Real},
    L::Real=500
)
    # Count number of unique barcodes
    n_bc = length(n_cells)

    # Compute the fraction each lineage (barcode) represents
    n_frac = n_cells ./ sum(n_cells)

    # Sample the number of cells to be used for PCR amplification
    return Random.rand.(Distributions.Poisson.(L * n_bc .* n_frac))
end # function

## ----------------------------------------------------------------------------- 

"""
    amplify_lineages(n_cells::Vector{<:Real}, n_cycles::Int=25) -> Matrix{Int}

Simulate the PCR amplification process for a set of lineage DNA samples over a
number of cycles.

# Arguments
- `n_cells::Vector{<:Real}`: Vector containing the starting number of DNA
  molecules for each lineage.
- `n_cycles::Int`: The number of PCR cycles to simulate. Defaults to 25 if not
  specified.

# Returns
`Matrix{Int}`: A matrix where each row corresponds to a lineage and each column
to a PCR cycle. Each element represents the number of DNA molecules for that
lineage at that cycle, after amplification.

# Description
This function models the PCR amplification process by iteratively doubling the
number of DNA molecules from the previous cycle, introducing variation with
Poisson noise at each step. The output is a matrix where each row represents a
lineage and columns represent successive PCR cycles.

# Example
```julia
amplified_DNA = amplify_lineages([10, 20, 30], 25)
```
"""
function amplify_lineages(
    n_cells::Vector{<:Real},
    n_cycles::Int=25
)
    # Initialize array to save outputs of each PCR cycle
    pcr_array = Matrix{Int64}(undef, length(n_cells), n_cycles + 1)

    # Set initial value
    pcr_array[:, 1] = n_cells

    # Loop through each PCR cyle
    for cycle = 2:(n_cycles+1)
        # Double previous time point with Poisson noise
        pcr_array[:, cycle] = Random.rand.(
            Distributions.Poisson.(2 .* pcr_array[:, cycle-1])
        )
    end # for

    return pcr_array
end # function

"""
    sequence_lineages(n_cells::Vector{<:Real}, reads::Int=20) -> Vector{Int}

Simulate sequencing noise for each lineage by Poisson sampling the number of
sequencing reads.

# Arguments
- `n_cells::Vector{<:Real}`: Vector containing the number of cells (or DNA
  fragments) for each lineage after PCR amplification.
- `reads::Int=20`: The scaling factor for the Poisson distribution, representing
  the average number of reads per cell.

# Returns
`Vector{Int}`: A vector containing the simulated sequencing read counts for each
lineage.

# Description
This function models the sequencing process, which introduces variability in the
number of reads that are actually observed for each lineage. The `reads`
parameter scales the Poisson distribution to reflect the average sequencing
coverage per cell or DNA fragment.

# Example
```julia
sequenced_reads = sequence_lineages([100, 200, 300])
```
"""
function sequence_lineages(
    n_cells::Vector{<:Real},
    reads::Int=20
)
    # Count number of unique barcodes
    n_bc = length(n_cells)

    # Compute the fraction each lineage (barcode) represents
    n_frac = n_cells ./ sum(n_cells)

    # Sample the number of cells to be used for PCR amplification
    return Random.rand.(Distributions.Poisson.(reads * n_bc .* n_frac))
end # function

## ----------------------------------------------------------------------------- 

"""
    fitseq2_fitness_measurement(
        λ̲::AbstractVector{Float64},
        n̲₀::AbstractVector{Int64};
        n_growth_cycles::Int64=4,
        n_pcr_cycles::Int64=25,
        n_gen::Int64=8,
        L::Real=500,
        reads::Int=100
    ) -> Matrix{Int64}

Simulate fitness measurement experiments over multiple growth-dilution cycles
with PCR amplification and sequencing, using specified growth rates for each
lineage.

# Arguments
- `λ̲::AbstractVector{Float64}`: Vector of growth rate values for each lineage.
- `n̲₀::AbstractVector{Int64}`: Vector with the initial population sizes for
  each lineage.

# Optional Keyword Arguments
- `n_growth_cycles::Int64=4`: Number of growth-dilution cycles to simulate.
- `n_pcr_cycles::Int64=25`: Number of PCR cycles to simulate.
- `n_gen::Int64=8`: Number of generations within each growth cycle.
- `L::Real=500`: Expected number of cells to sample from the culture for PCR
  amplification.
- `reads::Int=100`: Scaling factor for the Poisson distribution during
  sequencing, representing the average number of reads per lineage.

# Returns
- `Matrix{Int64}`: Matrix with the simulated sequencing read counts for each
  lineage across all growth-dilution cycles.

# Description
The function integrates multiple simulation steps across growth-dilution cycles:
1. Growth: Simulates noisy lineage growth over `n_gen` generations using
   associated fitness values.
2. Sampling: Poisson samples cells from the population for PCR, based on lineage
   abundance and expected sample size `L`.
3. PCR Amplification: Simulates noisy PCR amplification through `n_pcr_cycles`,
   introducing variability at each cycle.
4. Sequencing: Applies Poisson sampling to simulate sequencing noise,
   multiplying the number of cells by the `reads` parameter to simulate the
   average sequencing coverage.
5. Dilution: Prepares cultures for the next growth cycle by simulating the
   bottleneck effect of transferring cultures.

The simulation starts with the initial populations `n̲₀` and iteratively applies
these steps, storing the final read counts for each lineage at the end of each
cycle.

# Example
```julia
read_counts = fitseq2_fitness_measurement([0.05, 0.02], [500, 300])
```
"""
function fitseq2_fitness_measurement(
    λ̲::AbstractVector{<:Real},
    n̲₀::AbstractVector{<:Real};
    n_growth_cycles::Int64=4,
    n_pcr_cycles::Int64=25,
    n_gen::Int64=8,
    L::Real=500,
    reads::Int=100
)
    # Check the positivity of initial populations
    if any(n̲₀ .< 0)
        error("Initial populations cannot be negative")
    end # if

    # Define number of lineages
    n_lineage = length(n̲₀)

    # Check that every lineage has a fitness value assigned
    if length(λ̲) ≠ n_lineage
        error("Each lineage must have a corresponding fitness value")
    end # if

    # Initialize matrix to save number of cells through growth cycles
    n_cells_culture = Matrix{Float64}(undef, n_growth_cycles + 1, n_lineage)
    # Set initial condition
    n_cells_culture[1, :] = n̲₀

    # Initialize matrix to save number of sampled cells for PCR
    n_cells_sample = similar(n_cells_culture)
    # Set initial condition
    n_cells_sample[1, :] = sample_lineages(n_cells_culture[1, :], L)

    # Initialize matrix to save number of reads after PCR
    n_pcr = similar(n_cells_culture)
    # Set initial condition
    n_pcr[1, :] = amplify_lineages(n_cells_sample[1, :], n_pcr_cycles)[:, end]

    # Initialize matrix to save number of sequenced reads
    n_reads = similar(n_cells_culture)
    # Set initial condition
    n_reads[1, :] = sequence_lineages(n_pcr[1, :], reads)


    # Loop through growth-dilution cycles
    for cyc = 2:(n_growth_cycles+1)
        # 1. Simulate noisy growth for n_gen
        n_cells_culture[cyc, :] = growth_lineages(
            n_cells_culture[cyc-1, :], λ̲, n_gen
        )

        # 2. Simulate noisy cell sampling
        n_cells_sample[cyc, :] = sample_lineages(n_cells_culture[cyc, :], L)

        # 3. Simulate noisy PCR amplification
        n_pcr[cyc, :] = amplify_lineages(
            n_cells_sample[cyc, :], n_pcr_cycles
        )[:, end]

        # 4. Simulate noisy sequencing
        n_reads[cyc, :] = sequence_lineages(n_pcr[cyc, :], reads)

        # 5. Dilute back again the cultures for next generation
        n_cells_culture[cyc, :] = transfer_lineages(
            n_cells_culture[cyc, :], n_gen
        )
    end # for

    return n_reads
end # function