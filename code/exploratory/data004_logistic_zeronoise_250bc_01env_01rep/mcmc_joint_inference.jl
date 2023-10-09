##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

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

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.pboc_makie!()
##

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
    "$(git_root())/data/logistic_growth/data_004/tidy_data.csv", DF.DataFrame
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load prior inferences
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Group neutral data by barcode
data_group = DF.groupby(data[data.neutral, :], :barcode)

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
s_pop_prior = hcat(-logfreq_mean, repeat([0.05], length(logfreq_mean)))

# Generate single list of log-frequency ratios to compute prior on σ
logfreq_vec = vcat(logfreq...)

# Define priors for nuisance parameters for log-likelihood functions
σ_pop_prior = [-2.0, 0.1]
σ_mut_prior = σ_pop_prior

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
    :outputname => "./output/chain_joint_fitness_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
    :model => BarBay.model.fitness_lognormal,
    :model_kwargs => Dict(
        :s_pop_prior => s_pop_prior,
        :σ_pop_prior => σ_pop_prior,
        :σ_mut_prior => σ_mut_prior,
        :s_mut_prior => [0.0, 1.0],
    ),
    :sampler => Turing.externalsampler(DynamicHMC.NUTS()),
    :ensemble => Turing.MCMCSerial(),
    :rm_T0 => false,
)

# Run inference
println("Running Inference...")

@time BarBay.mcmc.mcmc_sample(; param...)