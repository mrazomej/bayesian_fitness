##
println("Loading packages...")



# Import project package
import BayesFitUtils

# Import library package
import BarBay

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

# Impor statistical libraries
import Random
import StatsBase
import Distributions

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
    "$(git_root())/data/logistic_growth/data_002/tidy_data.csv", DF.DataFrame
)

# Group data by repeat
df_group = DF.groupby(df, :rep)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load prior inferences
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Loop through groups
for (i, data) in enumerate(df_group)

    println("Loading prior information for R$i...\n")

    # List files
    chn_files = Glob.glob("./output/chain_popmean*jld2")

    # Load chain
    chn = JLD2.load(chn_files[i])["chain"]

    # Select variables for population mean fitness and associated variance
    var_name = MCMCChains.namesingroup.(Ref(chn), [:s̲ₜ, :σ̲ₜ])

    # Fit normal distributions to population mean fitness
    pop_mean = Distributions.fit.(
        Ref(Distributions.Normal), [vec(chn[x]) for x in var_name[1]]
    )

    # Fit lognormal distributions to associated error
    pop_std = Distributions.fit.(
        Ref(Distributions.LogNormal), [vec(chn[x]) for x in var_name[2]]
    )

    # Define parameters for population mean fitness
    s_pop_prior = hcat(
        first.(Distributions.params.(pop_mean)),
        last.(Distributions.params.(pop_mean))
    )
    # Define parameters for population mean fitness error
    σ_pop_prior = hcat(
        first.(Distributions.params.(pop_std)),
        last.(Distributions.params.(pop_std))
    )
    # Define parameters for mutant fitness error
    σ_mut_prior = maximum.(eachcol(σ_pop_prior))

    # List files
    chn_files = Glob.glob("./output/chain_freq*jld2")

    # Load chain into memory
    chn = JLD2.load(chn_files[i])["chain"]

    # Select variables for population mean fitness and associated variance
    var_name = MCMCChains.namesingroup(chn, :Λ̲̲)

    # Fit normal distributions to population mean fitness
    bc_freq = Distributions.fit.(
        Ref(Distributions.LogNormal), [vec(chn[x]) for x in var_name]
    )

    # Extract λ prior for frequencies
    λ_prior = hcat(
        first.(Distributions.params.(bc_freq)),
        last.(Distributions.params.(bc_freq))
    )

    println("Initializing MCMC sampling for R$i...\n")

    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => n_walkers,
        :n_steps => n_steps,
        :outputname => "./output/chain_joint_fitness_R$(i)rep_$(n_steps)steps_$(lpad(n_walkers, 2, "0"))walkers",
        :model => BarBay.model.fitness_lognormal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :σ_pop_prior => σ_pop_prior,
            :σ_mut_prior => σ_mut_prior,
            :s_mut_prior => [0.0, 1.0],
        ),
        :sampler => Turing.DynamicNUTS(),
        :ensemble => Turing.MCMCThreads(),
        :rm_T0 => false,
    )

    # Run inference
    println("Running Inference...")

    @time BarBay.mcmc.mcmc_joint_fitness(; param...)
end # for