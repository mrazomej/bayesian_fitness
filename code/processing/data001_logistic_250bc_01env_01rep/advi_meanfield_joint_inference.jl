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

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_001/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Obtain priors on expected errors from neutral measurements
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute naive priors from neutral strains
naive_priors = BayesFitness.stats.naive_prior(data)

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI function parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

param = Dict(
    :data => data,
    :outputname => "./output/advi_meanfield_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
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