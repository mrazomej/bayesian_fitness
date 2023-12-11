##
println("Loading packages...")



import Revise
import Suppressor
# Import project package
import BayesFitUtils
# Import library to list files
import Glob
# Import library to save output
import JLD2

# Import package to handle DataFrames
import DataFrames as DF
import CSV

# Import basic statistical functions
import StatsBase
import Random
import Distributions

# Import basic math
import LinearAlgebra

# Functionality for constructing arrays with identical elements efficiently
import FillArrays

# Import libraries relevant for MCMC
import Turing
import MCMCChains

Random.seed!(42)

##

# Define boolean on whether to suppress output
suppress_output = false
# Define Boolean on whether to remove T0
rm_T0 = false

## Define run parameters

# Number of steps
n_steps = 4_000
# Number of warm-up steps
n_warm = 200

# mutant fitness
# s⁽ᵐ⁾ ~ Normal(0, σₛ)
σₛ = 5.0
# Nuance parameter for mean fitness likelihood
# σ⁽ᵐ⁾ ~ Half-Normal(0, σₑ)
σₑ = 5.0

# Define population mean fitness directory
sₜ_dir = "./output/1000steps_4walkers"

##

println("Loading data...")

# Load data
df = CSV.read("$(git_root())/data/big_batch/tidy_counts.csv", DF.DataFrame)

# Remove T0 if indicated
if rm_T0
    println("Deleting T0 as requested...")
    df = df[.!(df.timepoint .== "T0"), :]
end # if

# List mean fitness files
mean_files = Glob.glob("$(sₜ_dir)/*jld2")

##

# Define output directory
if !ispath("./output")
    mkdir("./output")
end # if

# Define specific run output directory
out_dir = "./output/$(n_steps)steps_1walkers"
if !ispath(out_dir)
    mkdir(out_dir)
end # if

##

println("Writing down metadata to README.md file...")

# Define text to go into README
readme = """
# `$(@__FILE__)`

## Number of steps
n_steps = $(n_steps)
## Number of walkers
n_walkers = 1

## Mean fitness prior
    s⁽ᵐ⁾ ~ Normal(0, $(σₛ))
## Nuance parameter for mean fitness likelihood
    σ⁽ᵐ⁾ ~ Half-Normal($(σₑ))
"""

# Write README file into memory
open("$(out_dir)/README.md", "w") do file
    write(file, readme)
end

##

println("Defining Turing.jl model...")

Turing.@model function mutant_fitness(
    r̲⁽ᵐ⁾, r̲⁽ᶜ⁾, α̲, σₛ, σₑ, μ_sₜ, σ_sₜ
)
    # Prior on mutant fitness s⁽ᵐ⁾
    s⁽ᵐ⁾ ~ Turing.Normal(0.0, σₛ)
    # Prior on LogNormal error σ⁽ᵐ⁾ 
    σ⁽ᵐ⁾ ~ Turing.truncated(Turing.Normal(0.0, σₑ); lower=0.0)

    # Population mean fitness values
    s̲ₜ ~ Turing.MvNormal(μ_sₜ, LinearAlgebra.Diagonal(σ_sₜ .^ 2))

    # Initialize array to store frequencies
    f̲⁽ᵐ⁾ = Vector{Float64}(undef, length(r̲⁽ᵐ⁾))

    # Frequency distribution for each time point
    for i in eachindex(r̲⁽ᵐ⁾)
        f̲⁽ᵐ⁾[i] ~ Turing.Beta(α̲[1] + r̲⁽ᵐ⁾[i], α̲[2] + r̲⁽ᶜ⁾[i])
    end # for

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲⁽ᵐ⁾))
        Turing.@addlogprob! -Inf
        return
    end

    # Compute frequency ratios
    γ̲⁽ᵐ⁾ = f̲⁽ᵐ⁾[2:end] ./ f̲⁽ᵐ⁾[1:end-1]

    # Sample posterior for frequency ratio. Since it is a sample over a
    # generated quantity, we must use the @addlogprob! macro
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            s⁽ᵐ⁾ .- s̲ₜ,
            LinearAlgebra.I(length(s̲ₜ)) .* σ⁽ᵐ⁾^2
        ),
        γ̲⁽ᵐ⁾
    )

end # @model function

##

println("Grouping data...")
# Group data
df_group = DF.groupby(df, [:batch, :hub, :perturbation, :rep])
# Extract keys
df_keys = keys(df_group)

##

# Loop through groups
for (i, data) in enumerate(df_group)
    # Extract information
    batch, hub, perturbation, rep = collect(df_keys[i])

    # Define output file name
    fname = "$(batch)_$(hub)hub_$(perturbation)perturbation_$(rep)rep" *
            "_mutantfitness"

    # Check if this dataset has been ran before
    if isfile("$(out_dir)/$(fname)_mcmcchains.jld2")
        println("$fname was already processed")
        continue
    end # if

    println("Preparing $(fname)...")

    # Group data by unique mutant barcode
    data_group = DF.groupby(data[.!data.neutral, :], :barcode)

    # Extract keys
    data_keys = collect(keys(data_group))

    # Initialize array to save chains for each mutant
    chains = Array{MCMCChains.Chains}(undef, length(data_group))

    # Extract total number of barcodes per timepoint
    r_tot = DF.combine(DF.groupby(data, :timepoint), :count => sum)

    # Extract total number of unique barcodes
    n_tot = length(unique(data.barcode))

    # Initialize matrix to save counts for each mutant
    global r = Array{Float64}(
        undef, length(data_group), 2, length(unique(data.timepoint))
    )
    # Loop through each unique barcode
    for (i, d) in enumerate(data_group)
        # sort data by timepoint
        DF.sort!(d, :timepoint)
        # Extract data
        r[i, 1, :] = d.count
        r[i, 2, :] = r_tot.count_sum .- d.count
    end # for

    # Define alpha values
    α = [1, n_tot - 1]

    println("Loading mean fitness MCMC chains...")
    # Extract files for specific dataset
    global files = mean_files[
        occursin.(batch, mean_files).&occursin.(hub, mean_files).&occursin.("$(perturbation)perturbation", mean_files).&occursin.(rep, mean_files)
    ]

    # Initialize arrays to save mean and variance for population mean fitness
    µ_sₜ = Vector{Float64}(undef, length(files))
    σ_sₜ = similar(µ_sₜ)

    # Loop through files
    for (i, f) in enumerate(files)
        # Load MCMC chain
        chain = vec(Matrix(JLD2.load(f)["chain"][1][:, :sₜ, :]))
        # Fit Normal distribution
        dist = Distributions.fit(Distributions.Normal, chain)
        # Store mean and standard deviation
        µ_sₜ[i] = Distributions.mean(dist)
        σ_sₜ[i] = Distributions.std(dist)
    end # for

    # Loop through barcodes
    Threads.@threads for j = 1:size(r, 1)
        # Define model
        model = mutant_fitness(
            r[j, 1, :], r[j, 2, :], α, σₛ, σₑ, μ_sₜ, σ_sₜ
        )

        if suppress_output
            # Suppress warning outputs
            Suppressor.@suppress begin
                # Sample
                chains[j] = Turing.sample(
                    model,
                    Turing.NUTS(n_warm, 0.65),
                    n_steps,
                    progress=false
                )
            end # suppress
        else
            chains[j] = Turing.sample(
                model,
                Turing.NUTS(n_warm, 0.65),
                n_steps,
                progress=false
            )
        end # if
    end # for

    println("Saving $(fname) chains...")
    # Write output into memory
    JLD2.jldsave(
        "$(out_dir)/$(fname)_mcmcchains.jld2",
        chains=Dict(zip(data_keys, chains)),
        σₛ=σₛ,
        σₑ=σₑ
    )
    println("Done with $(fname)")

end # for