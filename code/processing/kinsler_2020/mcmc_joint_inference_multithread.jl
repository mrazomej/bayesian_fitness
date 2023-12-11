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

##

Random.seed!(42)

##

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# Define sampling hyperparameters
n_steps = 1000
n_walkers = 4

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate output directories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define output directory
outdir = "./output/joint_inference"

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

if !isdir(outdir)
    mkdir(outdir)
end # if

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading the data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
df = CSV.read("$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame)

# Read datasets visual evaluation info
df_include = CSV.read(
    "$(git_root())/data/kinsler_2020/exp_include.csv", DF.DataFrame
)

# NOTE: For the multithread to run when some files have already been processed,
# we need to avoid the "continue" condition when checking if a file was already
# processed (I honestly don't know why). Therefore, we must list all datasets to
# be processed, see which ones are complete, and remove them from the data fed
# to the sampler. This allows Threads.@threads to actually work.

# Initialize array indicating if dataset has been completely processed or not
complete_bool = repeat([false], size(df_include, 1))

# Loop through groups
for i in axes(df_include, 1)
    # Extract info
    env, rep, rm_T0 = collect(df_include[i, :])

    # Define output directory
    fname = "$(outdir)/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0.jld2"

    # Check if output doesn't exist
    if isfile(fname)
        # Idicate file is not complete
        complete_bool[i] = true
    end # if
end # for

println("Number of datasets previously processed: $(sum(complete_bool))")

# filtered datasets to process
df_include = df_include[.!(complete_bool), :]

println("$(size(df_include, 1)) datasets to process...\n")

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load prior inferences
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading prior information based on prior inferences...\n")

# Load prior on population mean fitness
df_popmean = CSV.read(
    "./output/popmean_fitness/popmean_fitness_priors.csv", DF.DataFrame
)

# Load prior on population mean fitness
df_freq = CSV.read(
    "./output/bc_freq/bc_freq_priors.csv", DF.DataFrame
)


##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Initialize MCMC sampling
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Initializing MCMC sampling...\n")

# Loop through datasets
for i in axes(df_include, 1)
    # Extract info
    env, rep, rm_T0 = collect(df_include[i, :])

    println("Processing $(env) | $(rep)")

    # Extract data
    data = df[(df.env.==env).&(df.rep.==rep), :]

    # Extract prior information
    popmean = df_popmean[(df_popmean.env.==env).&(df_popmean.rep.==rep), :]
    freq = df_freq[(df_freq.env.==env).&(df_freq.rep.==rep), :]

    # Define priors
    s_pop_prior = Matrix(popmean[:, [:s_mean, :s_std]])
    σ_pop_prior = Matrix(popmean[:, [:σ_mean, :σ_std]])
    σ_bc_prior = StatsBase.mean(eachcol(σ_pop_prior))

    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => n_walkers,
        :n_steps => n_steps,
        :outputname => "$(outdir)/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0_$(n_steps)steps_$(lpad(2, n_walkers, "0"))walkers",
        :model => BarBay.model.fitness_lognormal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :σ_pop_prior => σ_pop_prior,
            :σ_bc_prior => σ_bc_prior,
        ),
        :sampler => Turing.DynamicNUTS(),
        :ensemble => Turing.MCMCThreads(),
        :rm_T0 => rm_T0,
    )

    # Run inference
    println("Running Inference for group $(i)...")

    try
        @time BarBay.mcmc.mcmc_joint_fitness(; param...)
    catch
        @warn "Group $(i) was already processed"
    end
end # for
