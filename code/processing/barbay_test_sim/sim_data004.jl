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

# Define standard deviation for added Gaussian noise
σ_lognormal = 0.3
# Define ancestral strain growth rate
λ_a = 1.0
# Define carrying capacity
κ = 10.0^10
# Define number of generations
n_gen = 8
# Define number of neutral and mutants
n_neutral, n_mut = [5, 10]
# Define number of barcodes
n_bc = n_neutral + n_mut
# Define number of genotypes
n_geno = Int64(n_mut / 10)
# Define the number of barcodes per genotype
n_geno_bc = Random.rand(
    Distributions.Multinomial(n_mut, repeat([1 / n_geno], n_geno))
)

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

# Initialize array to store growth rates of genotypes
λ̲_geno = Vector{Float64}(undef, n_neutral + n_geno + 1)

# Set neutral growth rates
λ̲_geno[1:n_neutral+1] .= λ_a

# Define mutant fitness distribution mean
λ_bc_mean = λ_a * 1.005
# Define standard deviation to sample growth rates
λ_bc_std = 0.1

# Define truncation ranges for growth rates
λ_trunc = [λ_a .* 0.9999, λ_a * 1.5]

# Sample mutant growth rates
λ̲_geno[n_neutral+2:end] .= sort!(
    rand(
        Distributions.truncated(
            Distributions.Normal(λ_bc_mean, λ_bc_std), λ_trunc...
        ), n_geno
    )
)

# Initialize list to save all growth rates
λ̲ = Float64[]
# Add neutral growth rates
push!(λ̲, λ̲_geno[1:n_neutral+1]...)

# Loop through mutant barcodes
for (i, lam) in enumerate(λ̲_geno[n_neutral+2:end])
    # Add growth rate to list according to number of barcodes
    push!(λ̲, repeat([lam], n_geno_bc[i])...)
end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Simulate datasets 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Run deterministic simulation
n_mat = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲, n₀; n_gen=n_gen, κ=κ, σ_lognormal=0.0, poisson_noise=false
)
# Run noisy simulation
n_mat_noise = BayesFitUtils.sim.logistic_fitness_measurement(
    λ̲, n₀; n_gen=n_gen, κ=κ, σ_lognormal=σ_lognormal, poisson_noise=true
)

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
pop_mean_fitness = [
    StatsBase.mean(x[.!isinf.(x)])
    for x in eachrow(-logγ_mat[:, 1:n_neutral])
]

# Compute fitness by extracting the population mean fitness from the log
# frequency ratios and computing the mean of this quantity over time.
fitness = vec([
    StatsBase.mean(x[.!isinf.(x)] .+ pop_mean_fitness[.!isinf.(x)])
    for x in eachcol(logγ_mat)
])

# Create dataframe with relative fitness, growth rate, and genotype information
df_fit = DF.DataFrame(
    :barcode => bc_names,
    :fitness => fitness,
    :growth_rate => λ̲[2:end],
    :genotype => [
        repeat(["genotype000"], n_neutral)
        vcat([
            repeat(["genotype$(lpad(i, 3, "0"))"], n_geno_bc[i])
            for i in 1:n_geno
        ]...)
    ]
)

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
# Save data to memory
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 

# Defne output directory
out_dir = "$(homedir())/git/BarBay.jl/test/data"

CSV.write("$(out_dir)/data004_multigen.csv", data)