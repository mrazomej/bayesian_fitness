##
println("Loading packages...")



import Revise
# Import project package
import BayesFitUtils
# Import package for Bayesian inference
import BarBay

# Import differential equations package
using DifferentialEquations

# Import basic math
import StatsBase
import Distributions
import LinearAlgebra

# Import plotting libraries
using CairoMakie
import ColorSchemes
import ColorTypes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

# Import library to set random seed
import Random

Random.seed!(42)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define simulation parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of mutants
n_mut = 1000
# Define the number of neutrals
n_neutral = 50
# Define number of barcodes
n_bc = n_neutral + n_mut

# Define ancestral strain growth rate
λ_a = 1.0

# Define carrying capacity
κ = 10.0^10

# Define number of generations before saturation
n_gen = 8

# Define initial number of cells
n_init = κ / (2^(n_gen))

# Define initial number of cells for the rest of the mutants
n₀ = vcat([n_init * 0.9], repeat([n_init * 0.1 / n_mut], n_bc))

# Define fraction spent in exponential growth
exp_frac = 1 / 9

# Define time span
t_span = (0.0, log(2) / λ_a * n_gen / exp_frac)

# Initialize array to save growth rates
λ = Vector{Float64}(undef, n_bc + 1)

# Set neutral barcodes
λ[1:n_neutral+1] .= λ_a

# Define standard deviation to sample growth rates
σ_λ = 0.05

# Sample growth rates for all of the strains from a normal distribution centered
# at the ancestral strain growth rate
λ[n_neutral+2:end] = sort!(
    rand(
        Distributions.truncated(
            Distributions.Normal(λ_a * 1.005, σ_λ), λ_a * 0.9, λ_a * 1.2
        ), n_mut
    )
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Solve system of ODEs
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define ODE problem
prob = ODEProblem(BayesFitUtils.sim.dndt_rhs!, n₀, t_span, (λ, κ))

# Solve
sol = solve(prob)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of points
n_points = 200

# Define time range
t = LinRange(t_span[1], t_span[2], n_points)

# Evaluate interpolator
n = hcat(map(sol, t)...)

# Initialize figure
fig = Figure(size=(350, 300))

# Get colors for plot
colors = get(ColorSchemes.RdBu, λ[n_neutral+2:end] .- 1, :centered)

# Add axis
ax = Axis(fig[1, 1], xlabel="time (a.u.)", ylabel="# cells", yscale=log10)

# Loop through mutants
for (i, mut) in enumerate(eachrow(n[n_neutral+2:end, :]))
    # Plot trajectory
    lines!(ax, t, mut, color=colors[i])
end # for

# Plot neutral lineages
for (i, mut) in enumerate(eachrow(n[1:n_neutral, :]))
    # Plot trajectory
    lines!(ax, t, mut, color=:black, linestyle=:dash, linewidth=2)
end # for

save("$(git_root())/doc/figs/figSIX_logistic_growth_01.pdf", fig)
save("$(git_root())/doc/figs/figSIX_logistic_growth_01.png", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define a multi-cycle growth-dilution experiment
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of cycles to run
n_cycles = 5

# Inititalize matrix to save output
n_mat = Matrix{Float64}(undef, n_cycles + 1, n_bc + 1)

# Store initial condition
n_mat[1, :] .= n₀

# Loop through cycles
for cyc = 2:n_cycles+1
    # Define ODE problem
    local prob = ODEProblem(
        BayesFitUtils.sim.dndt_rhs!, n_mat[cyc-1, :], t_span, (λ, κ)
    )
    # Solve system and store final point
    n_mat[cyc, :] = solve(prob).u[end] ./ 2^(n_gen)
end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot final number of cells for multiple growth-dilution cycles
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(size=(350, 300))

# Add axis
ax = Axis(fig[1, 1], xlabel="cycle number", ylabel="# cells", yscale=log10)

# Loop through mutants
for (i, mut) in enumerate(eachcol(n_mat[:, n_neutral+2:end]))
    # Plot trajectory
    scatterlines!(ax, mut, markersize=8, color=(colors[i], 0.5))
end # for

# Loop through neutrals
for (i, neutral) in enumerate(eachcol(n_mat[:, 1:n_neutral+1]))
    # Plot trajectory
    scatterlines!(ax, neutral, markersize=8, color=:black, linestyle=:dash)
end # for

ylims!(ax, low=1)

save("$(git_root())/doc/figs/figSIX_logistic_growth_02.pdf", fig)
save("$(git_root())/doc/figs/figSIX_logistic_growth_02.png", fig)

fig


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot log-frequency ratios
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute the frequencies for all non-ancestral strains
f_mat = n_mat[:, 2:end] ./ sum(n_mat[:, 2:end], dims=2)

# Compute the frequency ratios
γ_mat = f_mat[2:end, :] ./ f_mat[1:end-1, :]

# Initialize figure
fig = Figure(size=(350, 300))

# Add axis
ax = Axis(fig[1, 1], xlabel="time [dilution cycles]", ylabel="ln(fₜ₊₁/fₜ)")

# Loop through mutants
for (i, mut) in enumerate(eachcol(γ_mat[:, n_neutral+2:end]))
    # Plot mutant trajectory
    scatterlines!(ax, log.(mut), color=(colors[i], 0.5), markersize=8)
end # for

# Loop through neutrals
for (i, neutral) in enumerate(eachcol(γ_mat[:, 1:n_neutral]))
    # Plot mutant trajectory
    scatterlines!(
        ax, log.(neutral), color=:black, markersize=8, linestyle=:dash
    )
end # for

save("$(git_root())/doc/figs/figSIX_logistic_growth_03.pdf", fig)
save("$(git_root())/doc/figs/figSIX_logistic_growth_03.png", fig)

fig


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Simulate growth-dilution experiment
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define ancestral strain growth rate
λ_a = 1.0
# Define carrying capacity
κ = 10.0^10
# Define number of generations
n_gen = 8
# Define number of neutral and mutants
n_neutral, n_mut = [25, 225]
# Define number of barcodes
n_bc = n_neutral + n_mut

# Compute initial number of cells
n_init = κ / (2^(n_gen))

# Define fracton of culture that is ancestor
frac_anc = 0.93
# Define fraction of culture that is neutrals
frac_neutral = 0.02
# Define fraction of culture that is mutants
frac_mut = 1 - frac_anc - frac_neutral

# Define initial number of cells
n₀ = [
    # ancestor
    Int64(round(n_init * frac_anc))
    # neutrals
    rand(
        Distributions.Poisson(n_init * frac_neutral / n_neutral),
        n_neutral
    )
    # mutants
    Int64.(
        round.(
            rand(
                Distributions.LogNormal(log(n_init * frac_mut / n_mut), 2),
                n_mut
            )
        )
    )
]

# Initialize array to store growth rates
λ̲ = Vector{Float64}(undef, n_bc + 1)

# Set neutral growth rates
λ̲[1:n_neutral+1] .= λ_a

# Define mutant fitness distribution mean
λ_mut_mean = λ_a * 1.005
# Define standard deviation to sample growth rates
λ_mut_std = 0.1

# Define truncation ranges for growth rates
λ_trunc = [λ_a .* 0.9999, λ_a * 1.5]

# Define level of Gaussian noise to add
σ_lognormal = 0.3

# Sample mutant growth rates
λ̲[n_neutral+2:end] .= sort!(
    rand(
        Distributions.truncated(
            Distributions.Normal(λ_mut_mean, λ_mut_std), λ_trunc...
        ), n_mut
    )
)

# Run deterministic simulation
n_mat = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲, n₀; n_gen=n_gen, κ=κ, σ_lognormal=0.0, poisson_noise=false
)
# Run noisy simulation
n_mat_noise = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲, n₀; n_gen=n_gen, κ=κ, σ_lognormal=σ_lognormal
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot deterministic and noisy trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute the frequencies for all non-ancestral strains
f_mat = (n_mat[:, 2:end] ./ sum(n_mat[:, 2:end], dims=2)) .+ 1E-9
f_mat_noise = (n_mat_noise[:, 2:end] ./ sum(n_mat_noise[:, 2:end], dims=2)) .+
              1E-9

# Compute the frequency ratios
γ_mat = f_mat[2:end, :] ./ f_mat[1:end-1, :]
γ_mat_noise = f_mat_noise[2:end, :] ./ f_mat_noise[1:end-1, :]

# Initialize figure
fig = Figure(size=(350 * 2, 300 * 2))


# Add axis
ax1 = [
    Axis(fig[i, 1], xlabel="time [dilution cycles]", ylabel="frequency", yscale=log10)
    for i = 1:2
]
ax2 = [
    Axis(fig[i, 2], xlabel="time [dilution cycles]", ylabel="ln(fₜ₊₁/fₜ)")
    for i = 1:2
]

# Define colors
color = ColorSchemes.glasbey_hv_n256

# Loop through mutants
for mut in (n_neutral+2):(n_bc)
    # Sample color
    col = (color[StatsBase.sample(1:length(color))], 0.25)
    # Plot mutant frequency trajectory
    scatterlines!(ax1[1], f_mat[:, mut], color=col, markersize=4)
    scatterlines!(ax1[2], f_mat_noise[:, mut], color=col, markersize=4)
    # Plot mutant log-frequency ratio trajectory
    scatterlines!(ax2[1], log.(γ_mat[:, mut]), color=col, markersize=4)
    scatterlines!(ax2[2], log.(γ_mat_noise[:, mut]), color=col, markersize=4)
end # for

# Loop through neutrals
for neutral in 1:n_neutral
    # Plot neutral frequency trajectory
    scatterlines!(
        ax1[1],
        f_mat[:, neutral],
        color=ColorSchemes.Blues_9[end],
        markersize=4
    )
    scatterlines!(
        ax1[2],
        f_mat_noise[:, neutral],
        color=ColorSchemes.Blues_9[end],
        markersize=4
    )

    # Plot mutant trajectory
    scatterlines!(
        ax2[1],
        log.(γ_mat[:, neutral]),
        color=ColorSchemes.Blues_9[end],
        markersize=4
    )
    scatterlines!(
        ax2[2],
        log.(γ_mat_noise[:, neutral]),
        color=ColorSchemes.Blues_9[end],
        markersize=4
    )
end # for

save("$(git_root())/doc/figs/figSIX_logistic_growth_04.pdf", fig)
save("$(git_root())/doc/figs/figSIX_logistic_growth_04.png", fig)

fig