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
import Distributions

Random.seed!(42)

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI hyerparameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples and steps
n_samples = 1
n_steps = 3_000

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate output directories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading the data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data..\n")

# Import data
data = CSV.read(
    "$(git_root())/data/abreu_2023/GHN_data.csv", DF.DataFrame
)

# Compute total reads per replicate and time point
df_sum = DF.combine(DF.groupby(data, [:replicate, :time]), :count => sum)

# Add total sum column to dataframe
DF.leftjoin!(data, df_sum; on=[:replicate, :time])

# Compute frequencies
data[!, :freq] = data[:, :count] ./ data[:, :count_sum]

# Define number of unique environments
n_env = length(unique(data.environment))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Obtain priors on expected errors from neutral measurements
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute naive priors from neutral strains
naive_priors = BayesFitness.stats.naive_prior(
    data; rep_col=:replicate, pseudocount=1
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

logσ_mut_prior = [StatsBase.mean(naive_priors[:logσ_pop_prior]), 1.0]

logλ_prior = hcat(
    naive_priors[:logλ_prior],
    repeat([3.0], length(naive_priors[:logλ_prior]))
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI function parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Preparing parameters for Variational Inference...\n")

# Define environment cycles
envs = collect(unique(data[:, [:time, :environment]])[:, :environment])

# Collect parameters in dictionary
param = Dict(
    :data => data,
    :outputname => "./output/advi_meanfield_hierarchical_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
    :model => BayesFitness.model.multienv_exprep_fitness_normal,
    :model_kwargs => Dict(
        :envs => envs,
        :s_pop_prior => s_pop_prior,
        :logσ_pop_prior => logσ_pop_prior,
        :logσ_mut_prior => logσ_mut_prior,
        :s_mut_prior => [0.0, 1.0],
        :logλ_prior => logλ_prior,
    ),
    :advi => Turing.ADVI(n_samples, n_steps),
    :opt => Turing.TruncatedADAGrad(),
    :rep_col => :replicate,
    :fullrank => false
)

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Convert output to tidy dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# List files
file = first(Glob.glob("./output/advi_meanfield_hier*"))

# Define environment cycles
envs = collect(
    unique(data[:, [:time, :environment]])[:, :environment]
)

# Load results
advi_results = JLD2.load(file)

# Extract components
mut_ids = advi_results["ids"]
dist = advi_results["dist"]
vars = advi_results["var"]

# Generate tidy dataframe with distribution information
df_advi = BayesFitness.utils.advi_to_df(
    dist, vars, mut_ids; envs=envs
)

# Save output
CSV.write(replace(file, ".jld2" => ".csv"), df_advi)