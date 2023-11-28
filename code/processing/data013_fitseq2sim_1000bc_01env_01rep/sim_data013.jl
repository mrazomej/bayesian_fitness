##
println("Loading packages...\n")
# Load project package
@load_pkg BayesFitUtils

import Revise
# Import project package
import BayesFitUtils

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
BayesFitUtils.viz.theme_makie!()

# Set random seed
Random.seed!(42)

## ----------------------------------------------------------------------------- 

# Define number of neutral lineages
n_neutral = 50
# Define total number of lineages
n_lin = 1_000
# Define number of mutants
n_mut = n_lin - n_neutral

# Define number of generations per cycle
n_gen = 4

# Define number of cycles
n_growth_cycle = 4

# Define average number of reads per barcode
reads = 1_000

# Define parameters for initial number of cells according to
# n₀ ~ Gamma(α, β)
α = 20
β = 0.2

# Define parameters for fitness distribution according to
# λ ~ SkewNormal(µ, σ, skew)
µ = 0.0
σ = 0.225
skew = 3

## ----------------------------------------------------------------------------- 

# Sample initial number of cells
n̲₀ = Random.rand(Distributions.Gamma(α, 1 / β), n_lin)# .* 100

# Sample fitness values
λ̲ = [
    repeat([0], n_neutral);
    sort(Random.rand(Distributions.SkewNormal(µ, σ, skew), n_mut))
] .+ log(2)

## ----------------------------------------------------------------------------- 

# Simulate experiment
n_mat_noise = BayesFitUtils.sim.fitseq2_simulation_noise(
    λ̲, n̲₀; n_growth_cycles=n_growth_cycle, n_gen=n_gen, reads=reads
)

# Compute true fitness
n_mat = BayesFitUtils.sim.fitseq2_simulation_noiseless(
    λ̲, n̲₀; n_growth_cycles=n_growth_cycle, n_gen=n_gen
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute frequencies and log-frequency ratios
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute the frequencies for all non-ancestral strains
f_mat = n_mat ./ sum(n_mat, dims=2) #.+ 1E-10

f_mat_noise = n_mat_noise ./ sum(n_mat_noise, dims=2) #.+ 1E-10

# Compute the frequency ratios
γ_mat = f_mat[2:end, :] ./ f_mat[1:end-1, :]
γ_mat_noise = f_mat_noise[2:end, :] ./ f_mat_noise[1:end-1, :]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Convert data to tidy dataframe
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
    :growth_rate => λ̲
)

# Convert matrix to dataframe
data = DF.DataFrame(n_mat_noise, bc_names)

# Add time column
data[!, :time] = 1:size(n_mat_noise, 1)

# Convert to tidy dataframe
data = DF.stack(data, bc_names)

# Rename columns
DF.rename!(data, :variable => :barcode, :value => :count)

# Convert noiseless matrix to dataframe
data_noiseless = DF.DataFrame(n_mat, bc_names)

# Add time column
data_noiseless[!, :time] = 1:size(n_mat, 1)

# Convert to tidy dataframe
data_noiseless = DF.stack(data_noiseless, bc_names)

# Rename columns
DF.rename!(data_noiseless, :variable => :barcode, :value => :count_noiseless)

# Join noiseless data and noise on barcode and time
DF.leftjoin!(data, data_noiseless; on=[:barcode, :time])

# Add neutral index column
data[!, :neutral] = occursin.("neutral", data.barcode)

# Build dataframe with count sum
data_sum = DF.combine(DF.groupby(data, :time), :count => sum)
DF.leftjoin!(data, data_sum; on=:time)

# Add frequency colymn
data[!, :freq] = data.count ./ data.count_sum

# Add fitness and growth rate information
DF.leftjoin!(data, df_fit; on=:barcode)


## ----------------------------------------------------------------------------- 

# Generate output directory if it doesn't exist
if !isdir("./output/")
    mkdir("./output/")
end # if

if !isdir("./output/figs")
    mkdir("./output/figs")
end # if

# Initialize figure
fig = Figure(resolution=(400, 500))

# Add axis
ax = [
    Axis(
        fig[i, 1],
        xlabel="time",
    ) for i = 1:2
]

# Plot mutant barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax[1],
    data[.!(data.neutral), :],
    zero_lim=1E-2,
    quant_col=:count,
    alpha=0.3,
    zero_label="extinct"
)

# Plot neutral barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax[1],
    data[data.neutral, :],
    zero_lim=1E-2,
    quant_col=:count,
    color=ColorSchemes.Blues_9[end],
    zero_label="extinct",
)

# Change scale
ax[1].yscale = log10
# Add label
ax[1].ylabel = "barcode count"

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

save("./output/figs/trajectories.pdf", fig)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 
# Save data to memory
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 

# Defne output directory
out_dir = "$(git_root())/data/logistic_growth/data_013"

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
# Number of generations
`n_gen = $(n_gen)`
"""

# Write README file into memory
open("$(out_dir)/README.md", "w") do file
    write(file, readme)
end

CSV.write("$(out_dir)/tidy_data.csv", data)