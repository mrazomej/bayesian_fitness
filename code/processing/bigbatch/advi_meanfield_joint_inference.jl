##
println("Loading packages...")
# Import project package
import BayesFitUtils

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to list files
import Glob

# Import library to perform Bayesian inference
import Turing
import DynamicHMC

# Import AutoDiff backend
using ReverseDiff
# Impor statistical libraries
import Random
import StatsBase



Random.seed!(42)

##

# Define number of samples and steps
n_samples = 1
n_steps = 10_000

##

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

# Load data
df = CSV.read("$(git_root())/data/big_batch/tidy_counts.csv", DF.DataFrame)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Perform ADVI on datasets
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Group data
df_group = DF.groupby(df, [:batch, :hub, :perturbation, :rep])
# Collect information
df_keys = values.(keys(df_group))

# Loop through datasets
Threads.@threads for i in 1:length(df_keys)

    # Extract metadata
    batch, hub, pert, rep = df_keys[i]

    println("Processing $(batch) | $(hub) | $(pert) | $(rep)")

    # Extract data
    data = df_group[i]

    # Compute naive parameters
    prior_param = BarBay.stats.naive_prior_neutral(data)

    # Define parameter for population mean fitness adding a standard deviation
    s_pop_prior = hcat(
        prior_param[:s_pop_prior],
        repeat([0.2], length(prior_param[:s_pop_prior]))
    )

    # Define function parameters
    param = Dict(
        :data => data,
        :outputname => "$(outdir)/bigbatch_$(batch)batch_$(hub)hub_" *
                       "$(pert)perturbation_$(rep)rep_" *
                       "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
        :model => BarBay.model.fitness_normal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :logσ_pop_prior => prior_param[:logσ_pop_prior],
            :logσ_bc_prior => prior_param[:logσ_pop_prior],
            :s_bc_prior => [0.0, 1.0],
        ),
        :advi => Turing.ADVI(n_samples, n_steps),
        :opt => Turing.TruncatedADAGrad(),
        :fullrank => false,
        :rm_T0 => false
    )

    # Run inference
    println("Running Inference for group $(i) / $(length(df_group))...")

    @time BarBay.vi.advi(; param...)
end # for