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

# Import plotting libraries
using CairoMakie
import ColorSchemes

import Random

Random.seed!(42)

##

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

println("Loading data...")

# Import data
df = CSV.read("$(git_root())/data/kinsler_2020/tidy_counts.csv", DF.DataFrame)

##

# NOTE: For the multithread to run when some files have already been processed,
# we need to avoid the "continue" condition when checking if a file was already
# processed (I honestly don't know why). Therefore, we must list all datasets to
# be processed, see which ones are complete, and remove them from the data fed
# to the sampler. This allows Threads.@threads to actually work.

# Group data by env, rep
df_group = DF.groupby(df, [:env, :rep])

# Extract keys
df_keys = collect(keys(df_group))

# Initialize array indicating if dataset has been completely processed or not
complete_bool = Vector{Bool}(undef, length(df_keys))

# Loop through groups
for (i, data) in enumerate(df_group)
    # Extract group info
    env, rep = [df_keys[i]...]

    # Define output directory
    outdir = "./output/$(env)_$(rep)"

    # Check if output doesn't exist
    if !isdir(outdir)
        # Idicate file is not complete
        complete_bool[i] = false

        # Check number of files in directory matches number of time points
    elseif length(readdir(outdir)) != length(unique(data.time)) - 1
        # Idicate file is not complete
        complete_bool[i] = false

        # Check file number matches
    elseif length(readdir(outdir)) == length(unique(data.time)) - 1
        # Idicate file is complete
        complete_bool[i] = true
    end # if
end # for

println("Number of datasets previously processed: $(sum(complete_bool))")

# Initialize dataframe to use
df_filt = DF.DataFrame()

# Loop through groups
for data in df_group[.!complete_bool]
    DF.append!(df_filt, data)
end # for

##

# Group data by environment
df_group = DF.groupby(df_filt, [:env, :rep])

##

# Extract group keys
df_keys = collect(keys(df_group))

println("Initializing MCMC sampling...\n")

# Loop through datasets
Threads.@threads for i = 1:length(df_group)
    # Extract data
    data = df_group[i]

    # Extract keys
    env, rep = df_keys[i][1], df_keys[i][2]

    # define outputdir
    outdir = "./output/$(env)_$(rep)"
    # Remove existing directory
    rm(outdir, recursive=true, force=true)

    # Make output dir if necessary
    if !isdir(outdir)
        mkdir(outdir)
    end # if

    println("Processing $(env)-$(rep) mean fitness ($i / $(length(df_group))) \n")


    # Define parameters
    param = Dict(
        :data => data,
        :n_walkers => 4,
        :n_steps => 4_000,
        :outputdir => outdir,
        :outputname => "$(env)_$(rep)_meanfitness",
        :model => BayesFitness.model.mean_fitness_neutrals_lognormal,
        :model_kwargs => Dict(
            :α => BayesFitness.stats.dirichlet_prior_neutral(
                data[data.time.==0, :neutral],
            )
        ),
        :id_col => :barcode,
        :time_col => :time,
        :count_col => :count,
        :neutral_col => :neutral,
        :multithread => false
    )

    # Run inference
    BayesFitness.mcmc.mcmc_mean_fitness(; param...)
end # for

##