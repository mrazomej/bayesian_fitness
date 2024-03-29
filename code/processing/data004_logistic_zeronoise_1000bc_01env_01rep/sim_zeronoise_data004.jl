##

println("Loading packages...\n")
# Load project package
@load_pkg BayesFitUtils

import Revise
# Import project package
import BayesFitUtils

# Import differential equations package
using DifferentialEquations

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import basic statistical functions
import StatsBase
import Distributions
import Random

# Load CairoMakie for plotting
using CairoMakie
import ColorSchemes
import Makie
# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.pboc_makie!()

# Set random seed
Random.seed!(42)

# Define if plots should be generated
gen_plots = true

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define simulated experiment parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define standard deviation for added Gaussian noise
σ_lognormal = 0.0
# Define ancestral strain growth rate
λ_a = 1.0
# Define carrying capacity
κ = 10.0^10
# Define number of generations
n_gen = 8
# Define number of neutral and mutants
n_neutral, n_mut = [100, 900]
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

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define random initial number of cels
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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
λ_bc_mean = λ_a * 1.005
# Define standard deviation to sample growth rates
λ_bc_std = 0.1

# Define truncation ranges for growth rates
λ_trunc = [λ_a .* 0.9999, λ_a * 1.5]

# Sample mutant growth rates
λ̲[n_neutral+2:end] .= sort!(
    rand(
        Distributions.truncated(
            Distributions.Normal(λ_bc_mean, λ_bc_std), λ_trunc...
        ), n_mut
    )
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Simulate datasets 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Run deterministic simulation
n_mat = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲, n₀; n_gen=n_gen, κ=κ, σ_lognormal=0.0, poisson_noise=false
)
# Run noisy simulation
n_mat_noise = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲, n₀; n_gen=n_gen, κ=κ, σ_lognormal=σ_lognormal, poisson_noise=false
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute frequencies and log-frequency ratios
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute the frequencies for all non-ancestral strains
f_mat = n_mat[:, 2:end] ./ sum(n_mat[:, 2:end], dims=2) .+ 1E-9

f_mat_noise = n_mat_noise[:, 2:end] ./ sum(n_mat_noise[:, 2:end], dims=2) .+ 1E-9

# Compute the frequency ratios
γ_mat = f_mat[2:end, :] ./ f_mat[1:end-1, :]
γ_mat_noise = f_mat_noise[2:end, :] ./ f_mat_noise[1:end-1, :]

##
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute deterministic fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define barcode name
bc_names = [
    ["neutral$(lpad(i, 3,  "0"))" for i = 1:n_neutral]
    ["mut$(lpad(i, 3,  "0"))" for i = 1:n_mut]
]

# Take the log of the frequency ratios
logγ_mat = log.(γ_mat)

# Obtain population mean fitness given the neutrals
pop_mean_fitness = StatsBase.mean(-logγ_mat[:, 1:n_neutral], dims=2)

# Compute fitness by extracting the population mean fitness from the log
# frequency ratios and computing the mean of this quantity over time.
fitness = vec(StatsBase.mean(logγ_mat .- pop_mean_fitness, dims=1))

# Create dataframe with relative fitness and growth rate
df_fit = DF.DataFrame(
    :barcode => bc_names,
    :fitness => fitness .- StatsBase.mean(fitness[1:n_neutral]),
    :growth_rate => λ̲[2:end]
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Convert data to tidy dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Convert matrix to dataframe
data = DF.DataFrame(n_mat_noise[:, 2:end], bc_names)

# Add time column
data[!, :time] = 1:size(n_mat_noise, 1)

# Convert to tidy dataframe
data = DF.stack(data, bc_names)

# Rename columns
DF.rename!(data, :variable => :barcode, :value => :count)

# Add neutral index column
data[!, :neutral] = occursin.("neutral", data.barcode)

# Build dataframe with count sum
data_sum = DF.combine(DF.groupby(data, :time), :count => sum)
DF.leftjoin!(data, data_sum; on=:time)

# Add frequency colymn
data[!, :freq] = data.count ./ data.count_sum

# Add fitness and growth rate information
DF.leftjoin!(data, df_fit; on=:barcode)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot barcode trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate output directory if it doesn't exist
if !isdir("./output/")
    mkdir("./output/")
end # if

if !isdir("./output/figs")
    mkdir("./output/figs")
end # if


if gen_plots
    # Initialize figure
    fig = Figure(resolution=(400, 500))

    # Add axis
    ax = [
        Axis(
            fig[i, 1],
            xlabel="time",
        ) for i = 1:2
    ]

    # %%% Barcode trajectories %%% #

    # Plot mutant barcode trajectories
    BayesFitUtils.viz.bc_time_series!(
        ax[1],
        data[.!(data.neutral), :],
        zero_lim=0,
        alpha=0.3
    )

    # Plot neutral barcode trajectories
    BayesFitUtils.viz.bc_time_series!(
        ax[1],
        data[data.neutral, :],
        zero_lim=0,
        color=ColorSchemes.Blues_9[end],
    )

    # Change scale
    ax[1].yscale = log10
    # Add label
    ax[1].ylabel = "barcode frequency"

    # %%% log-frequency ratio trajectories %%% #

    # Plot mutant barcode trajectories
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax[2],
        data[.!(data.neutral), :],
        alpha=0.3
    )

    # Plot neutral barcode trajectories
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax[2],
        data[data.neutral, :],
        color=ColorSchemes.Blues_9[end],
    )

    # Add label
    ax[2].ylabel = "ln(fₜ₊₁/fₜ)"
end # if

save("./output/figs/trajectories.pdf", fig)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 
# Save data to memory
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 

# Defne output directory
out_dir = "$(git_root())/data/logistic_growth/data_004"

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Number of mutant barcodes
`n_mut = $(n_mut)`
## number of neutral barcodes
`n_neutral = $(n_neutral)`
## Ancestral strain growth rate
`λ_a = $(λ_a)`
## Carrying capacity
`κ = $(κ)`
# Number of generations
`n_gen = $(n_gen)`
## Initial number of cells
n_init = κ / (2^(n_gen)) = $(n_init)
## Initial fracton of culture that is ancestor
`frac_anc = $(frac_anc)`
## Initial fraction of culture that is neutrals
`frac_neutral = $(frac_neutral)`
## Initial fraction of culture that is mutants
`frac_mut = $(frac_mut)`
## Mutant fitness distribution mean
`λ_bc_mean = λ_a * 1.005 = $(λ_bc_mean)`
## Mutant fitness distribution standard deviation to sample growth rates
`λ_bc_std = $(λ_bc_std)`
## Mutant fitness distribution truncation ranges for growth rates
`λ_trunc = [λ_a .* 0.9999, λ_a * 1.5] = $(λ_trunc)`
## Gaussian noise distribution standard deviation
`σ_lognormal = $(σ_lognormal)`
"""

# Write README file into memory
open("$(out_dir)/README.md", "w") do file
    write(file, readme)
end

CSV.write("$(out_dir)/tidy_data.csv", data)