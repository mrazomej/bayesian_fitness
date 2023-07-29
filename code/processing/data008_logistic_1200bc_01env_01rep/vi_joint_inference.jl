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

Random.seed!(42)

# Set AutoDiff backend
Turing.setadbackend(:reversediff)
# Allow system to generate cache to speed up computation
Turing.setrdcache(true)

##

# Define if plots should be generated
gen_plots = false

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI hyerparameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples and steps
n_samples = 1
n_steps = 10_000

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
    "$(git_root())/data/logistic_growth/data_008/tidy_data.csv", DF.DataFrame
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Obtain priors on expected errors from neutral measurements
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Group neutral data by barcode
data_group = DF.groupby(data[data.neutral, :], :barcode)

# Initialize list to save log frequency changes
logfreq = []

# Loop through each neutral barcode
for d in data_group
    # Sort data by time
    DF.sort!(d, :time)
    # Compute log frequency ratio and append to list
    push!(logfreq, diff(log.(d[:, :freq])))
end # for

# Generate matrix with log-freq ratios
logfreq_mat = hcat(logfreq...)

# Compute mean per time point for approximate mean fitness
logfreq_mean = StatsBase.mean(logfreq_mat, dims=2)

# Define prior for population mean fitness
s_pop_prior = hcat(-logfreq_mean, repeat([0.3], length(logfreq_mean)))

# Generate single list of log-frequency ratios to compute prior on σ
logfreq_vec = vcat(logfreq...)

# Define priors for nuisance parameters for log-likelihood functions
σ_pop_prior = [StatsBase.mean(logfreq_vec), StatsBase.std(logfreq_vec)]
σ_mut_prior = σ_pop_prior

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI function parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

param = Dict(
    :data => data,
    :outputname => "./output/advi_meanfield_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
    :model => BayesFitness.model.fitness_normal,
    :model_kwargs => Dict(
        :s_pop_prior => s_pop_prior,
        :σ_pop_prior => σ_pop_prior,
        :σ_mut_prior => σ_mut_prior,
        :s_mut_prior => [0.0, 1.0],
    ),
    :advi => Turing.ADVI(n_samples, n_steps),
    :opt => Turing.TruncatedADAGrad(),
    :fullrank => false
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Perform optimization
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Variational Inference...")
@time dist = BayesFitness.vi.vi_joint_fitness(; param...)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load advi distribution and corresponding MCMC chain
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

if gen_plots
    # Load MCMC results
    ids_chn, chn = values(
        JLD2.load("./output/chain_joint_fitness_1000steps_04walkers.jld2")
    )

    # Load ADVI results
    ids_advi, advi = values(
        JLD2.load("./output/advi_meanfield_$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps.jld2")
    )
end # if

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare inferences
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

if gen_plots
    # Locate fitness values in first model
    s_idx = occursin.("s̲⁽", String.(names(chn)))

    # Obtain variable names
    var_names = names(chn)[s_idx]

    # Initialize matrix to save parameters
    infer_param = Matrix{Float64}(undef, length(var_names), 2)

    # Loop through each chain element
    for (i, v) in enumerate(var_names)
        # Extract chain and compute mean and std
        infer_param[i, :] = [StatsBase.mean(chn[v]), StatsBase.std(chn[v])]
    end # for

    # Extract ADVI distribution parameters
    advi_param = Distributions.params(advi)

    # Extract relevant values
    infer_param = hcat(
        [
            infer_param,
            advi_param[1][s_idx[1:end-1]],
            advi_param[2][s_idx[1:end-1]]
        ]...
    )

    # Convert to tidy dataframe
    df_param = DF.DataFrame(
        infer_param, ["mcmc_mean", "mcmc_std", "advi_mean", "advi_std"]
    )
    # Add mutant information
    DF.insert_single_column!(df_param, var_names, "mutant")

    ##

    # Initialize figure
    fig = Figure(resolution=(350, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="ADVI inference",
        ylabel="MCMC inference",
        title="fitness comparison",
        aspect=AxisAspect(1)
    )

    # Plot identity line
    lines!(ax, [-0.5, 1.75], [-0.5, 1.75], linestyle=:dash, color="black")

    # Plot x-axis error bars
    errorbars!(
        ax,
        df_param.advi_mean,
        df_param.mcmc_mean,
        df_param.advi_std,
        df_param.advi_std,
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.5)
    )
    # Plot y-axis error bars
    errorbars!(
        ax,
        df_param.advi_mean,
        df_param.mcmc_mean,
        df_param.mcmc_std,
        df_param.mcmc_std,
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.5)
    )

    # Plot fitness values
    scatter!(ax, df_param.advi_mean, df_param.mcmc_mean, markersize=5)

end # if