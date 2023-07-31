##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import project package
import BayesFitUtils

# Import library package
import BayesFitness

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

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

Random.seed!(42)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI hyerparameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples and steps
n_samples = 1
n_steps = 10_000

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
# Obtain priors on expected errors from neutral measurements
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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
logσ_pop_prior = vcat(
    [
        hcat(repeat([x], length(unique(data.time)) - 1)...)' for x in logσ_pop_p
    ]...
)
logσ_mut_prior = vcat(
    [
        hcat(repeat([x],
            length(unique(data[.!data.neutral, :barcode])))...)'
        for x in logσ_pop_p
    ]...
)
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI function parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

param = Dict(
    :data => data,
    :outputname => "./output/advi_meanfield_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
    :model => BayesFitness.model.exprep_fitness_normal,
    :model_kwargs => Dict(
        :s_pop_prior => s_pop_prior,
        :logσ_pop_prior => logσ_pop_prior,
        :logσ_mut_prior => logσ_mut_prior,
        :s_mut_prior => [0.0, 1.0],
    ),
    :advi => Turing.ADVI(n_samples, n_steps),
    :opt => Turing.TruncatedADAGrad(),
    :rep_col => :rep,
    :fullrank => false
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Perform optimization
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Variational Inference...")
@time dist = BayesFitness.vi.advi(; param...)