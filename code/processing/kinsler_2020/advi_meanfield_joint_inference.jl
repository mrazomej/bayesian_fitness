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

# Import library to list files
import Glob

# Import library to perform Bayesian inference
import Turing

# Import AutoDiff backend
using ReverseDiff

# Import Memoization
using Memoization

# Impor statistical libraries
import Random
import StatsBase

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

Random.seed!(42)

##

# Define number of samples and steps
n_samples = 1
n_steps = 4_500

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate output directories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define output directory
outdir = "./output/advi_meanfield_joint_inference"

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
df = CSV.read(
    "$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame
)

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

    # Define output file
    fname = "$(outdir)/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0_" *
            "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps.jld2"

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Perform ADVI on datasets
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Loop through datasets
Threads.@threads for i in axes(df_include, 1)
    # Extract info
    env, rep, rm_T0 = collect(df_include[i, :])

    println("Processing $(env) | $(rep)")

    # Extract data
    data = df[(df.env.==env).&(df.rep.==rep), :]

    # Compute naive priors from neutral strains
    naive_priors = BayesFitness.stats.naive_prior(data; rm_T0=rm_T0)

    # Select standard deviation parameters
    s_pop_prior = hcat(
        naive_priors[:s_pop_prior],
        repeat([0.05], length(naive_priors[:s_pop_prior]))
    )

    logσ_pop_prior = hcat(
        naive_priors[:logσ_pop_prior],
        repeat([1.0], length(naive_priors[:logσ_pop_prior]))
    )

    logσ_bc_prior = [StatsBase.mean(naive_priors[:logσ_pop_prior]), 1.0]

    logλ_prior = hcat(
        naive_priors[:logλ_prior],
        repeat([3.0], length(naive_priors[:logλ_prior]))
    )

    # Define function parameters
    param = Dict(
        :data => data,
        :outputname => "$(outdir)/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0_" *
                       "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
        :model => BayesFitness.model.fitness_normal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :logσ_pop_prior => logσ_pop_prior,
            :logσ_bc_prior => logσ_bc_prior,
            :s_bc_prior => [0.0, 1.0],
            :logλ_prior => logλ_prior,
        ),
        :advi => Turing.ADVI(n_samples, n_steps),
        :opt => Turing.TruncatedADAGrad(),
        :fullrank => false,
        :rm_T0 => rm_T0
    )

    # Run inference
    println("Running Inference for group $(i)...")

    @time BayesFitness.vi.advi(; param...)
end # for