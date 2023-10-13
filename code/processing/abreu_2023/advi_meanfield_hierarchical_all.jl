##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

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
n_steps = 4_500

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
df_counts = CSV.read(
    "$(git_root())/data/abreu_2023/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Process individual datasets
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Select columns that indicate the dataset
df_idx = df_counts[:, 13:end]

# Enumerate the columns (needed to run things in parallel smoothly)
n_col = size(df_idx, 2)

# Loop through each dataset
Threads.@threads for col = 1:n_col
    # Select data
    data = df_counts[df_idx[:, col], 1:12]

    # Extract information
    condition = first(data.condition)

    println("Performing inference for $(condition) / $(names(df_idx)[col]) | ($(col) / $(n_col))")

    # Define output directory
    out_dir = "./output/$(condition)_$(names(df_idx)[col])"

    # Generate output directory
    if !isdir(out_dir)
        mkdir(out_dir)
    end # if

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Define list of environments
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Define number of environments
    n_env = first(data.n_env)
    println("$(out_dir) #environments = $(n_env)")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Obtain priors on expected errors from neutral measurements
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Define ADVI function parameters
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    println("Preparing parameters for Variational Inference...\n")

    if n_env == 1
        # Collect parameters in dictionary
        param = Dict(
            :data => data,
            :outputname => "$(out_dir)/advi_meanfield_hierarchical_" *
                           "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
            :model => BarBay.model.replicate_fitness_normal,
            :model_kwargs => Dict(
                :s_pop_prior => s_pop_prior,
                :logσ_pop_prior => logσ_pop_prior,
                :logσ_bc_prior => logσ_bc_prior,
                :s_bc_prior => [0.0, 1.0],
                :logλ_prior => logλ_prior,
            ),
            :advi => Turing.ADVI(n_samples, n_steps),
            :opt => Turing.TruncatedADAGrad(),
            :rep_col => :rep,
            :fullrank => false
        )
    elseif n_env > 1
        # Collect parameters in dictionary
        param = Dict(
            :data => data,
            :outputname => "$(out_dir)/advi_meanfield_hierarchical_" *
                           "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
            :model => BarBay.model.multienv_replicate_fitness_normal,
            :model_kwargs => Dict(
                :s_pop_prior => s_pop_prior,
                :logσ_pop_prior => logσ_pop_prior,
                :logσ_bc_prior => logσ_bc_prior,
                :s_bc_prior => [0.0, 1.0],
                :logλ_prior => logλ_prior,
            ),
            :advi => Turing.ADVI(n_samples, n_steps),
            :opt => Turing.TruncatedADAGrad(),
            :rep_col => :rep,
            :env_col => :env,
            :fullrank => false
        )
    end # if
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Perform optimization
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Run inference
    println("Running Variational Inference for $(condition)...")
    @time dist = BarBay.vi.advi(; param...)

end # for
