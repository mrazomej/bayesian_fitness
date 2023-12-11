##
println("Loading packages...")



# Import project package
import BayesFitUtils

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

# Import library to perform Bayesian inference
import Turing
import MCMCChains
import DynamicHMC

# Import AutoDiff backend
using ReverseDiff

# Import Memoization
using Memoization

# Impor statistical libraries
import Random
import StatsBase
import Distributions

Random.seed!(42)

##

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# Define sampling hyperparameters
n_steps = 1
n_walkers = 4

# Define if plots should be generated
gen_plots = false

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate output directories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading the data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_002/tidy_data.csv", DF.DataFrame
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load prior inferences
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Defining priors from neutral lineages data...\n")

# Initialize list to save priors
s_pop_p = []
logσ_pop_p = []
logσ_mut_p = []

# Group data by replicates
data_rep = DF.groupby(data[data.neutral, :], :rep)

# Loop through replicates
for df in data_rep
    # Group neutral data by barcode
    data_group = DF.groupby(df, :barcode)

    # Initialize list to save log frequency changes
    logfreq = []

    # Loop through each neutral barcode
    for d in data_group
        # Sort data by time
        DF.sort!(d, :time)
        # Compute log frequency ratio and append to list
        push!(logfreq, diff(log.(d[:, :freq])))
    end # for

    # Generate matrix with log-freq ratios
    logfreq_mat = hcat(logfreq...)

    # Compute mean per time point for approximate mean fitness
    logfreq_mean = StatsBase.mean(logfreq_mat, dims=2)

    # Define prior for population mean fitness
    push!(
        s_pop_p, hcat(-logfreq_mean, repeat([0.3], length(logfreq_mean)))
    )

    # Generate single list of log-frequency ratios to compute prior on σ
    logfreq_vec = vcat(logfreq...)

    # Define priors for nuisance parameters for log-likelihood functions
    push!(
        logσ_pop_p, [StatsBase.mean(logfreq_vec), StatsBase.std(logfreq_vec)]
    )
    push!(
        logσ_mut_p, [StatsBase.mean(logfreq_vec), StatsBase.std(logfreq_vec)]
    )
end # for

# Convert priors to long matrices with repeated values to give unique priors to
# each replicate
s_pop_prior = vcat(s_pop_p...)
σ_pop_prior = vcat(
    [
        hcat(repeat([x], length(unique(data.time)) - 1)...)' for x in logσ_pop_p
    ]...
)
σ_mut_prior = vcat(
    [
        hcat(repeat([x],
            length(unique(data[.!data.neutral, :barcode])))...)'
        for x in logσ_pop_p
    ]...
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Initialize MCMC sampling
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Initializing MCMC sampling...\n")

# Define function parameters
param = Dict(
    :data => data,
    :n_walkers => n_walkers,
    :n_steps => n_steps,
    :outputname => "./output/chain_joint_hierarchical_fitness_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BarBay.model.exprep_fitness_lognormal,
    :model_kwargs => Dict(
        :s_pop_prior => s_pop_prior,
        :σ_pop_prior => σ_pop_prior,
        :σ_mut_prior => σ_mut_prior,
        :s_mut_prior => [0.0, 1.0],
    ),
    :rep_col => :rep,
    :sampler => Turing.externalsampler(DynamicHMC.NUTS()),
    :ensemble => Turing.MCMCThreads(),
    :rm_T0 => false,
)

# Run inference
println("Running Inference...")

@time BarBay.mcmc.mcmc_sample(; param...)
