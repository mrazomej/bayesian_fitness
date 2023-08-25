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

# Import library to save and load native julia objects
import JLD2

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI hyerparameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples and steps
n_samples = 1
n_steps = 10_000

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
    "$(git_root())/data/logistic_growth/data_009/tidy_data.csv", DF.DataFrame
)

# Define number of unique environments
n_env = length(unique(data.env))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Obtain priors on expected errors from neutral measurements
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Group data by replicate
data_rep = DF.groupby(data, :rep)

# Loop through replicates
Threads.@threads for rep = 1:length(data_rep)
    println("Running inference for replicate $rep")
    # Define environment cycles
    envs = collect(
        unique(data_rep[rep][:, [:time, :env]])[:, :env]
    )

    # Compute naive parameters
    prior_param = BayesFitness.stats.naive_prior_neutral(data_rep[rep])

    # Define parameter for population mean fitness adding a standard deviation
    s_pop_prior = hcat(
        prior_param[:s_pop_prior],
        repeat([0.2], length(prior_param[:s_pop_prior]))
    )

    # Extract counts as fed to inference pipeline
    mat_counts = BayesFitness.utils.data2arrays(data_rep[rep])[:bc_count]

    # Set priors for λ parameter
    logλ_prior = hcat(
        log.(mat_counts[:] .+ 1), repeat([2.0], length(mat_counts))
    )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Define ADVI function parameters
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    param = Dict(
        :data => data_rep[rep],
        :outputname => "./output/advi_meanfield_R$(rep)_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
        :model => BayesFitness.model.multienv_fitness_normal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :logσ_pop_prior => prior_param[:logσ_pop_prior],
            :logσ_mut_prior => prior_param[:logσ_pop_prior],
            :s_mut_prior => [0.0, 1.0],
            :logλ_prior => logλ_prior,
            :envs => envs,
        ),
        :advi => Turing.ADVI(n_samples, n_steps),
        :opt => Turing.TruncatedADAGrad(),
        :fullrank => false
    )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Perform optimization
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Run inference
    println("Running Variational Inference...")
    @time dist = BayesFitness.vi.advi(; param...)

end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Convert output to tidy dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# List files
files = Glob.glob("./output/advi_meanfield_R*10000*jld2")

# Define environment cycles
envs = collect(
    unique(data[:, [:time, :env]])[:, :env]
)

# Loop through files
for f in files
    # Load results
    advi_results = JLD2.load(f)

    # Extract components
    mut_ids = advi_results["ids"]
    dist = advi_results["dist"]
    vars = advi_results["var"]

    # Generate tidy dataframe with distribution information
    global df_advi = BayesFitness.utils.advi2df(
        dist, vars, mut_ids; n_rep=1, envs=envs
    )

    # Save output
    CSV.write(replace(f, ".jld2" => ".csv"), df_advi)
end # for