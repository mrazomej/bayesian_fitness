##

println("Loading packages...\n")


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

# Set random seed
Random.seed!(42)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define simulated experiment parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of environments
n_env = 3
# Define standard deviation for added Gaussian noise
σ_lognormal = 0.125
# Define ancestral strain growth rate
λ_a = [1.0, 2.0, 0.5]
# Define carrying capacity
κ = 10.0^10
# Define number of generations
n_gen = 8
# Define number of neutral and mutants
n_neutral, n_mut = [5, 10]
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

# Define standard deviation to sample growth rates
λ_bc_std = 0.1

# Initialize array to store growth rates
λ̲̲ = Matrix{Float64}(undef, n_bc + 1, n_env)

# Loop through environments
for env in 1:n_env
    # Set neutral growth rates
    λ̲̲[1:n_neutral+1, env] .= λ_a[env]

    # Define mutant fitness distribution mean
    λ_bc_mean = λ_a[env] * 1.005

    # Define truncation ranges for growth rates
    λ_trunc = [λ_a[env] .* 0.9995, λ_a[env] * 1.5]

    # Sample mutant growth rates
    λ̲̲[n_neutral+2:end, env] .= sort!(
        rand(
            Distributions.truncated(
                Distributions.Normal(λ_bc_mean, λ_bc_std), λ_trunc...
            ), n_mut
        )
    )
end # for
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Simulate datasets 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define environment cycles
env_idx = [1, 2, 3, 1, 2, 3]

# Run deterministic simulation
n_mat = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲̲,
    n₀,
    env_idx;
    n_gen=n_gen,
    κ=κ,
    σ_lognormal=0.0,
    poisson_noise=false
)

# Run noisy simulation
n_mat_noise = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲̲,
    n₀,
    env_idx;
    n_gen=n_gen,
    κ=κ,
    σ_lognormal=σ_lognormal
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

# # Initializ dataframe to save deterministic fitness
df_fit = DF.DataFrame()

# Loop through datasets
for env in 1:n_env
    # Take the log of the frequency ratios
    logγ_mat = log.(γ_mat[env_idx.==env, :])

    # Obtain population mean fitness given the neutrals
    pop_mean_fitness = StatsBase.mean(-logγ_mat[:, 1:n_neutral], dims=2)

    # Compute fitness by extracting the population mean fitness from the log
    # frequency ratios and computing the mean of this quantity over time.
    fitness = vec(StatsBase.mean(logγ_mat .- pop_mean_fitness, dims=1))

    # Create dataframe with relative fitness and growth rate
    DF.append!(
        df_fit,
        DF.DataFrame(
            :barcode => bc_names,
            :fitness => fitness .- StatsBase.mean(fitness[1:n_neutral]),
            :growth_rate => λ̲̲[2:end, env],
            :env .=> env
        )
    )
end # for

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Convert data to tidy dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Convert matrix to dataframe
data = DF.DataFrame(n_mat_noise[:, 2:end], bc_names)

# Add time column
data[!, :time] = 1:size(n_mat_noise, 1)
# Add environment column
data[!, :env] = [[1]; env_idx]

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
DF.leftjoin!(data, df_fit; on=[:barcode, :env])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 
# Save data to memory
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 

# Defne output directory
out_dir = "$(homedir())/git/BarBay.jl/test/data"

CSV.write("$(out_dir)/data003_multienv.csv", data)