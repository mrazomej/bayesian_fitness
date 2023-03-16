##
println("Loading packages...")

# Load project package
@load_pkg BayesFitness

import Revise
import Suppressor
# Import project package
import BayesFitness
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
n_steps = 1_000
# Number of warm-up steps
n_warm = 500

# Number of walkers
n_walkers = Threads.nthreads()

# Mean fitness prior
# s̲ₜ ~ Normal(0, σ_st)
σ_st = 1.5
# Nuance parameter for mean fitness likelihood
# σ̲ₘ ~ Half-Normal(σ_σt)
σ_σt = 1.0

##

println("Loading data...")

# Load data
df = CSV.read("$(git_root())/data/big_batch/tidy_counts.csv", DF.DataFrame)

# Remove T0 if indicated
if rm_T0
    println("Deleting T0 as requested...")
    df = df[.!(df.timepoint .== "T0"), :]
end # if

##

# Define output directory
if !ispath("./output")
    mkdir("./output")
end # if

# Define specific run output directory
out_dir = "./output/$(n_steps)steps_$(n_walkers)walkers"
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
n_walkers = $(n_walkers)

## Mean fitness prior
    sₜ ~ Normal(0, $(σ_st))
## Nuance parameter for mean fitness likelihood
    σₜ ~ Half-Normal($(σ_σt))
"""

# Write README file into memory
open("$(out_dir)/README.md", "w") do file
    write(file, readme)
end

##

println("Defining Turing.jl model...")

Turing.@model function mean_fitness_neutrals(r̲ₜ, r̲ₜ₊₁, α̲, σₛ, σₑ)
    # Prior on mean fitness sₜ
    sₜ ~ Turing.Normal(0.0, σₛ)
    # Prior on LogNormal error σₜ
    σₜ ~ Turing.truncated(Turing.Normal(0.0, σₑ); lower=0.0)

    # Frequency distribution from Multinomial-Dirichlet model
    f̲ₜ ~ Turing.Dirichlet(α̲ .+ r̲ₜ)
    f̲ₜ₊₁ ~ Turing.Dirichlet(α̲ .+ r̲ₜ₊₁)

    # Check that all distributions are greater than zero. Although the counts
    # could be zero, we assume that the real frequencies are non-zero always.
    if any(iszero.(f̲ₜ)) | any(iszero.(f̲ₜ₊₁))
        Turing.@addlogprob! -Inf
        return
    end

    # Compute frequency ratio
    γₜ = (f̲ₜ₊₁./f̲ₜ)[1:end-1]

    # Sample posterior for frequency ratio
    Turing.@addlogprob! Turing.logpdf(
        Turing.MvLogNormal(
            FillArrays.Fill(-sₜ, length(γₜ)),
            LinearAlgebra.I(length(γₜ)) .* σₜ^2
        ),
        γₜ
    )

end # @model function

##

println("Grouping data...")
# Group data
df_group = DF.groupby(df, [:batch, :hub, :perturbation, :rep])
# Extract keys
df_keys = keys(df_group)

# Loop through groups
for (i, d) in enumerate(df_group)
    # Copy data
    data_ = DF.DataFrame(deepcopy(d))
    # Extract information
    batch, hub, perturbation, rep = collect(df_keys[i])
    # Extract unique timepoints 
    timepoints = sort(unique(data_.timepoint))
    # Loop through pairs of timepoints
    for t = 1:(length(timepoints)-1)
        # Define output file name
        fname = "$(batch)_$(hub)hub_$(perturbation)perturbation_$(rep)rep_" *
                "$(timepoints[t])-$(timepoints[t+1])_meanfitness"

        # Check if this dataset has been ran before
        if isfile("$(out_dir)/$(fname)_mcmcchains.jld2")
            println("$fname was already processed")
            continue
        end # if

        println("Preparing $(fname)...")

        # Select correspoinding data
        data = data_[
            (data_.timepoint.==timepoints[t]).|(data_.timepoint.==timepoints[t+1]),
            :]

        # Group data by neutral barcode
        data_group = DF.groupby(data[data.neutral, :], :barcode)

        # Initialize array to save counts
        r = Matrix{Int64}(undef, length(data_group) + 1, 2)

        # Loop through barcodes
        for (i, group) in enumerate(data_group)
            # Sort data by timepoint
            DF.sort!(group, :timepoint)
            r[i, :] = group.count
        end # for

        # Add mutant counts
        r[end, 1] = sum(
            data[.!(data.neutral).&(data.timepoint.==timepoints[t]), :count]
        )
        r[end, 2] = sum(
            data[.!(data.neutral).&(data.timepoint.==timepoints[t+1]), :count]
        )

        # Set α values for Dirichlet distribution
        α = ones(size(r, 1))
        # Modify last α value for all mutant phenotypes
        α[end] = sum(.!(data.neutral) .& (data.timepoint .== timepoints[t]))

        # Define model
        model = mean_fitness_neutrals(r[:, 1], r[:, 2], α, σ_st, σ_σt)

        # Initialize object where to save chains
        chain = Vector{MCMCChains.Chains}(undef, 1)

        println("Sampling $(fname)...")
        if suppress_output
            # Suppress warning outputs
            Suppressor.@suppress begin
                # Sample
                chain[1] = Turing.sample(
                    model,
                    Turing.NUTS(n_warm, 0.65),
                    Turing.MCMCThreads(),
                    n_steps,
                    n_walkers,
                    progress=false
                )
            end # suppress
        else
            chain[1] = Turing.sample(
                model,
                Turing.NUTS(n_warm, 0.65),
                Turing.MCMCThreads(),
                n_steps,
                n_walkers,
                progress=true
            )
        end # if

        println("Saving $(fname) chains...")
        # Write output into memory
        JLD2.jldsave(
            "$(out_dir)/$(fname)_mcmcchains.jld2",
            chain=chain,
            σ_st=σ_st,
            σ_σt=σ_σt
        )

        println("Done with $(fname)")
    end # for
end # for
println("Done!")