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

# Group data by replicate
data_rep = DF.groupby(data, :replicate)

# Loop through replicates
Threads.@threads for rep = 1:length(data_rep)
    println("Running inference for replicate $rep")
    # Define environment cycles
    envs = collect(
        unique(data_rep[rep][:, [:time, :environment]])[:, :environment]
    )

    # Compute naive priors from neutral strains
    naive_priors = BayesFitness.stats.naive_prior(
        data_rep[rep]; pseudocount=1
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Define ADVI function parameters
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    param = Dict(
        :data => data_rep[rep],
        :outputname => "./output/advi_meanfield_R$(rep)_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
        :model => BayesFitness.model.multienv_fitness_normal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :logσ_pop_prior => logσ_pop_prior,
            :logσ_bc_prior => logσ_bc_prior,
            :s_bc_prior => [0.0, 1.0],
            :logλ_prior => logλ_prior,
            :envs => envs,
        ),
        :advi => Turing.ADVI(n_samples, n_steps),
        :opt => Turing.TruncatedADAGrad(),
        :fullrank => false
    )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Perform optimization
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Run inference
    println("Running Variational Inference...")
    @time dist = BayesFitness.vi.advi(; param...)

end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Convert output to tidy dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# List files
files = Glob.glob("./output/advi_meanfield_R*")

# Define environment cycles
envs = collect(
    unique(data[:, [:time, :environment]])[:, :environment]
)

# Loop through files
for f in files
    # Load results
    advi_results = JLD2.load(f)

    # Extract components
    bc_ids = advi_results["ids"]
    dist = advi_results["dist"]
    vars = advi_results["var"]

    # Generate tidy dataframe with distribution information
    df_advi = BayesFitness.utils.advi_to_df(
        dist, vars, bc_ids; envs=envs
    )

    # Save output
    CSV.write(replace(f, ".jld2" => ".csv"), df_advi)
end # for