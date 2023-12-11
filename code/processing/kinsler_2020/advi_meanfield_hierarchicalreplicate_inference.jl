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
n_steps = 10_000

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate output directories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define output directory
outdir = "./output/advi_meanfield_hierarchicalreplicate_inference"

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

if !isdir(outdir)
    mkdir(outdir)
end # if


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading the data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
df_counts = CSV.read(
    "$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame
)

# Read datasets visual evaluation info
df_include = CSV.read(
    "$(git_root())/data/kinsler_2020/exp_include.csv", DF.DataFrame
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Extract datasets with more than one replicate
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Group by replicate
df_group = DF.groupby(df_include, :env)

# Locate environments with more than one replicate to be included
rep_envs = [first(d.env) for d in df_group if (size(d, 1) > 1)]

# Loop through environments
Threads.@threads for i = 1:length(rep_envs)
    # Extract environment
    env = rep_envs[i]

    println("Processing environment: $env")
    # Extract include information
    df_in = df_include[df_include.env.==env, :]

    # Initialize dataframe to save data
    data = DF.DataFrame()

    # Loop through datasets
    for row in eachrow(df_in)
        # Extract dataset
        d = df_counts[(df_counts.env.==row.env).&(df_counts.rep.==row.rep), :]
        # Check if first time point should be removed
        if row.rm_T0
            # Remove first time point if required in df_include
            d = d[d.time.>minimum(d.time), :]
        end # if
        # Append dataframes
        DF.append!(data, d)
    end # for

    # Compute naive priors from neutral strains
    naive_priors = BarBay.stats.naive_prior(
        data; rep_col=:rep, pseudocount=1
    )

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

    # Define ADVI function parameters
    param = Dict(
        :data => data,
        :outputname => "$(outdir)/advi_meanfield_hierarchical_$(env)" *
                       "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
        :model => BarBay.model.replicate_fitness_normal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :logσ_pop_prior => logσ_pop_prior,
            :logσ_bc_prior => logσ_bc_prior,
            :s_bc_prior => [0.0, 1.0],
            :logλ_prior => logλ_prior,
            :logτ_prior => [-2.0, 0.5],
        ),
        :advi => Turing.ADVI(n_samples, n_steps),
        :opt => Turing.TruncatedADAGrad(),
        :rep_col => :rep,
        :fullrank => false
    )

    # Run inference
    println("Running Variational Inference...")
    @time BarBay.vi.advi(; param...)
end # for