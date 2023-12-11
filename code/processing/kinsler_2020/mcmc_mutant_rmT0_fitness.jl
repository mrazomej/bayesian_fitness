##
println("Loading packages...")



# Import package to revise package
import Revise
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

# Remove T0 data
df = df[df.time.!=0, :]

# Group data by environment
df_group = DF.groupby(df, [:env, :rep])

##

# Extract group keys
df_keys = collect(keys(df_group))

println("Initializing MCMC sampling...\n")

# Loop through datasets
for i = 1:length(df_group)
    # Extract data
    data = df_group[i]

    # Extract keys
    env, rep = df_keys[i][1], df_keys[i][2]

    # define outputdir
    outdir = "./output/$(env)_$(rep)"

    # Define pattern for population mean fitness samples
    meanfit_pattern = "$(env)_$(rep)_meanfitness"

    # Extract meanfitness files
    mean_files = Glob.glob("$(outdir)/$(meanfit_pattern)*")

    # Remove T0 file
    mean_files = mean_files[.!occursin.("0-1.jld", mean_files)]

    # Check that there are mean_fitness files
    if length(mean_files) == 0
        println("$(meanfit_pattern) files cannot be found")
        # Skip iteration
        continue
    elseif length(mean_files) != length(unique(data.time)) - 1
        println("Missing mean fitness files for $(meanfit_pattern)")
        # Skip iteration
        continue
    end # if

    # Infer mean fitness distributions
    mean_fitness_dist = BarBay.stats.gaussian_prior_mean_fitness(
        BarBay.utils.var_jld2_to_df(sort!(mean_files), :sₜ)
    )

    println("Processing $(env)-$(rep) mutant fitness ($i / $(length(df_group))) \n")

    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => 4,
        :n_steps => 4_000,
        :outputdir => outdir,
        :outputname => "$(env)_$(rep)_mutantfitness_rmT0",
        :model => BarBay.model.mutant_fitness_lognormal,
        :model_kwargs => Dict(
            :α => BarBay.stats.beta_prior_mutant(
                data[data.time.==minimum(data.time), :barcode],
            ),
            :μ_s̄ => mean_fitness_dist[1],
            :σ_s̄ => mean_fitness_dist[2],
        ),
        :multithread_mutant => true,
    )

    # Run inference in a multithread fasshion
    BarBay.mcmc.mcmc_mutant_fitness(; param...)

end # for