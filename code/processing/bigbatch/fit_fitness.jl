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

# Import basic math
import LinearAlgebra

# Import libraries relevant for MCMC
import Turing
import MCMCChains

Random.seed!(42)

##

# Define boolean on whether to suppress output
suppress_output = false
# Define Boolean on whether to remove T0
rm_T0 = false
# Define boolean indicating if specific datasets should be fit
specific_fit = true

# Define parameters to run specific datasets
batch = ["Batch2"]
hub = ["1Day"]
perturbation = ["1.5"]
rep = ["R1", "R2"]


## Define run parameters

# Number of steps
n_steps = 1_000
# Number of walkers
n_walkers = Threads.nthreads()

# Mutant fitness prior
# s̲ₘ ~ Normal(0, σ_sm)
σ_sm = Float32(1.5)
# Nuance parameter for mutant fitness likelihood
# σ̲ₘ ~ Half-Normal(σ_σm)
σ_σm = Float32(1.0)

# Mean fitness prior
# s̲ₜ ~ Normal(0, σ_st)
σ_st = Float32(1.5)
# Nuance parameter for mean fitness likelihood
# σ̲ₘ ~ Half-Normal(σ_σt)
σ_σt = Float32(1.0)

##

println("Loading data...")

# Load data
df = CSV.read("$(git_root())/data/big_batch/tidy_counts.csv", DF.DataFrame)

# Remove T0 if indicated
if rm_T0
    println("Deleting T0 as requested...")
    df = df[.!(df.timepoint .== "T0"), :]
end # if

# Keep specified data
if specific_fit
    println("Filtering data as requested...")
    df = df[
        ([x ∈ batch for x in df.batch]).&([x ∈ hub for x in df.hub]).&([x ∈ perturbation for x in df.perturbation]).&([x ∈ rep for x in df.rep]),
        :]
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

## Mutant fitness prior
    s̲ₘ ~ Normal(0, $(σ_sm))
## Nuance parameter for mutant fitness likelihood
    σ̲ₘ ~ Half-Normal($(σ_σm))

## Mean fitness prior
    s̲ₜ ~ Normal(0, $(σ_st))
## Nuance parameter for mean fitness likelihood
    σ̲ₘ ~ Half-Normal($(σ_σt))
"""

# Write README file into memory
open("$(out_dir)/README.md", "w") do file
    write(file, readme)
end

##

println("Defining Turing.jl model...")

Turing.@model function fitness_mutants(
    logf_mut::Matrix{Float32},
    logf_neutral::Matrix{Float32},
    t::Vector{Float32},
    σ_sm::Float32,
    σ_σm::Float32,
    σ_st::Float32,
    σ_σt::Float32
)
    # Define parameter types
    s̲ₘ = Float32[]  # adaptive mutants fitness
    σ̲ₘ = Float32[]  # variance for adaptive mutants fitness likelihood

    s̲ₜ = Float32[]  # population mean fitness vector
    σ̲ₜ = Float32[]  # variance for population mean fitness likelihood

    ## Prior

    # Prior on adaptive mutants mean fitness
    s̲ₘ ~ Turing.filldist(Turing.Normal(0, σ_sm), size(logf_mut, 2))
    # Prior on variance for adaptive mutants fitness likelihood
    σ̲ₘ ~ Turing.filldist(
        Turing.truncated(Turing.Cauchy(0, σ_σm); lower=0.0 + eps()),
        size(logf_mut, 2)
    )

    # Prior on the mean fitness
    s̲ₜ ~ Turing.filldist(Turing.Normal(0, σ_st), length(t))
    # Prior on error 
    σ̲ₜ ~ Turing.filldist(
        Turing.truncated(Turing.Cauchy(0, σ_σt); lower=0.0 + eps()),
        length(t)
    )

    ## Likelihood

    # Population mean fitness

    # Loop through neutral lineages
    for j = 1:size(logf_neutral, 2)
        # Define non-infinity indexes
        idx = .!isinf.(logf_neutral[:, j])
        # Sample
        logf_neutral[idx, j] ~ Turing.MvNormal(
            -s̲ₜ[idx], LinearAlgebra.Diagonal(σ̲ₜ[idx] .^ 2)
        )
    end # for

    # Adaptive mutants fitness

    # Loop through mutants lineages
    for j = 1:size(logf_mut, 2)
        # Define non-infinity indexes
        idx = .!isinf.(logf_mut[:, j])
        # Sample
        logf_mut[idx, j] ~ Turing.MvNormal(
            s̲ₘ[j] .- s̲ₜ[idx],
            LinearAlgebra.I(sum(idx)) .* (σ̲ₘ[j] .^ 2)
        )
    end # for

    return logf_mut, logf_neutral
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
    data = DF.DataFrame(deepcopy(d))
    # Extract information
    batch, hub, perturbation, rep = collect(df_keys[i])


    # Define output file name
    fname = "$(batch)_$(hub)_$(perturbation)_$(rep)"

    # Check if this dataset has been ran before
    if isfile("$(out_dir)/$(fname)_mcmcchains.jld2")
        println(
            "$(hub)hub_$(batch)batch_$(perturbation)perturbation_$(rep)rep" *
            "was already processed"
        )
        continue
    end # if

    println("Preparing $(fname)...")

    # Extract total counts per barcode
    data_total = DF.combine(DF.groupby(data, :time), :count => sum)
    # Add total count column to dataframe
    DF.leftjoin!(data, data_total; on=:time)
    # Add frequency column
    DF.insertcols!(data, :freq => data.count ./ data.count_sum)

    # Initialize dataframe to save the log freq changes
    data_log = DF.DataFrame()

    # Group data by barcode
    data_group = DF.groupby(data, :barcode)

    # Loop through each group
    for d in data_group
        # Compute log change
        DF.append!(
            data_log,
            DF.DataFrame(
                :barcode .=> first(d.barcode),
                :time => d.time[2:end],
                :logf => diff(log.(d.freq)),
                :neutral .=> first(d.neutral)
            )
        )
    end # for

    # Extract data for mutants
    logf_mut = hcat(
        [
            Float32.(DF.sort(d, :time).logf)
            for d in DF.groupby(data_log[.!data_log.neutral, :], :barcode)
        ]...
    )
    # Extract data for neutral lineages
    logf_neutral = hcat(
        [
            Float32.(DF.sort(d, :time).logf)
            for d in DF.groupby(data_log[data_log.neutral, :], :barcode)
        ]...
    )

    # Define model
    model = fitness_mutants(
        logf_mut,
        logf_neutral,
        Float32.(unique(data_log.time)),
        σ_sm,
        σ_σm,
        σ_st,
        σ_σt
    )

    # Initialize object where to save chains
    chain = Vector{MCMCChains.Chains}(undef, 1)

    println("Sampling $(fname)...")
    if suppress_output
        # Suppress warning outputs
        Suppressor.@suppress begin
            # Sample
            chain[1] = Turing.sample(
                model,
                Turing.NUTS(0.65),
                Turing.MCMCThreads(),
                n_steps,
                n_walkers,
                progress=false
            )
        end # suppress
    else
        chain[1] = Turing.sample(
            model,
            Turing.NUTS(0.65),
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
        σ_sm=σ_sm,
        σ_σm=σ_σm,
        σ_st=σ_st,
        σ_σt=σ_σt
    )

    println("Done with $(fname)")
end # for

println("Done!")