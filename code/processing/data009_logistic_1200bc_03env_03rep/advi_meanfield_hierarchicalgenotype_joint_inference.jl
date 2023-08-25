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
n_steps = 50_000

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
    "$(git_root())/data/logistic_growth/data_009/tidy_data.csv", DF.DataFrame
)

# Define number of unique environments
n_env = length(unique(data.env))

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Obtain priors on expected errors from neutral measurements
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
    # Compute naive priors
    naive_prior = BayesFitness.stats.naive_prior_neutral(df; pseudocount=1)

    # Define prior for population mean fitness
    push!(
        s_pop_p,
        hcat(
            naive_prior[:s_pop_prior],
            repeat([0.1], length(naive_prior[:s_pop_prior]))
        )
    )

    # Define priors for nuisance parameters for log-likelihood functions
    push!(
        logσ_pop_p, naive_prior[:logσ_pop_prior]
    )
    push!(
        logσ_mut_p, naive_prior[:logσ_pop_prior]
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
            length(unique(data[.!data.neutral, :barcode])) * n_env)...)'
        for x in logσ_pop_p
    ]...
)

# Extract counts as fed to inference pipeline
mat_counts = BayesFitness.utils.data2arrays(data; rep_col=:rep)[:bc_count]

# Set priors for λ parameter
logλ_prior = hcat(log.(mat_counts[:] .+ 1), repeat([2.0], length(mat_counts)))

##
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI function parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define environment cycles
env_idx = [1, 1, 2, 3, 1, 2, 3]

# Collect parameters in dictionary
param = Dict(
    :data => data,
    :outputname => "./output/advi_meanfield_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
    :model => BayesFitness.model.multienv_exprep_fitness_normal,
    :model_kwargs => Dict(
        :envs => env_idx,
        :s_pop_prior => s_pop_prior,
        :logσ_pop_prior => logσ_pop_prior,
        :logσ_mut_prior => logσ_mut_prior,
        :s_mut_prior => [0.0, 1.0],
        :logλ_prior => logλ_prior,
    ),
    :advi => Turing.ADVI(n_samples, n_steps),
    :opt => Turing.TruncatedADAGrad(),
    :rep_col => :rep,
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