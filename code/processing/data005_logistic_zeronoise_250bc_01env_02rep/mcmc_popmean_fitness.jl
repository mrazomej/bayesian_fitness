##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import package to revise package
import Revise
# Import project package
import BayesFitUtils

# Import library package
import BayesFitness

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import library to save and load native julia objects
import JLD2

# Import library to list files
import Glob

# Import library to perform Bayesian inference
import Turing
import MCMCChains
import DynamicHMC

# Import AutoDiff backend
using ReverseDiff

# Import Memoization
using Memoization

# Impor library to set random seed
import Random

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.pboc_makie!()

##

Random.seed!(42)

##

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# Define sampling hyperparameters
n_steps = 1000
n_walkers = 4

# Define if plots should be generated
gen_plots = false

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
df = CSV.read(
    "$(git_root())/data/logistic_growth/data_005/tidy_data.csv", DF.DataFrame
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Initialize MCMC sampling
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Initializing MCMC sampling...\n")

# Group data by repeat
df_group = DF.groupby(df, :rep)

# Loop through repeats
for (rep, data) in enumerate(df_group)
    println("Sampling for repeat R$(rep)...\n")
    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => n_walkers,
        :n_steps => n_steps,
        :outputname => "./output/chain_popmean_fitness_R$(rep)rep_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
        :model => BayesFitness.model.neutrals_lognormal,
        :sampler => Turing.DynamicNUTS(),
        :ensemble => Turing.MCMCThreads(),
        :rm_T0 => false,
    )

    # Run inference
    println("Running Inference...")
    @time BayesFitness.mcmc.mcmc_popmean_fitness(; param...)

end # for

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate diagnostic plots
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #