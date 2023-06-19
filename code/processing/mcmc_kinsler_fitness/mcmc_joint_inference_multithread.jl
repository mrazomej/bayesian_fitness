#

println("Importing packages...\n")
# Activate environment
@load_pkg BayesFitUtils

# Import library package
import BayesFitness

# Import MCMC-related packages
import Turing
using ReverseDiff

# Set AutoDiff backend to ReverseDiff.jl for faster computation
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

import Random

Random.seed!(42)

# Define inference parameters
n_walkers = 4
n_steps = 1000

##

# Import data
df = CSV.read(
    "$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame
)

# Read datasets visual evaluation info
df_include = CSV.read(
    "$(git_root())/data/kinsler_2020/exp_include.csv", DF.DataFrame
)

# Loop through datasets
for i = 1:size(df_include, 1)
    # Extract info
    env, rep, rm_T0 = collect(df_include[i, :])
    # Extract data
    data = df[(df.env.==env).&(df.rep.==rep), :]

    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => n_walkers,
        :n_steps => n_steps,
        :outputname => "./output/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0",
        :model => BayesFitness.model.fitness_lognormal,
        :sampler => Turing.NUTS(0.65),
        :ensemble => Turing.MCMCThreads(),
        :rm_T0 => rm_T0,
    )

    # Run inference
    println("Running Inference for group $(i)...")

    try
        @time BayesFitness.mcmc.mcmc_joint_fitness(; param...)
    catch
        @warn "Group $(i) was already processed"
    end
end # for
