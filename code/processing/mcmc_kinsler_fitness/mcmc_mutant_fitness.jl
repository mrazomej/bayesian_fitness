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
    # Check that there are mean_fitness files
    if length(mean_files) == 0
        println("$(meanfit_pattern) files cannot be found")
        # Skip iteration
        continue
    end # if

    # Infer mean fitness distributions
    mean_fitness_dist = BayesFitness.stats.gaussian_prior_mean_fitness(
        BayesFitness.utils.var_jld2_to_df(outdir, meanfit_pattern, :sₜ)
    )

    println("Processing $(env)-$(rep) mutant fitness ($i / $(length(df_group))) \n")


    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => 4,
        :n_steps => 2_500,
        :outputdir => outdir,
        :outputname => "$(env)_$(rep)_mutantfitness",
        :model => BayesFitness.model.mutant_fitness_lognormal,
        :model_kwargs => Dict(
            :α => BayesFitness.stats.beta_prior_mutant(
                data[data.time.==0, :barcode],
            ),
            :μ_s̄ => mean_fitness_dist[1],
            :σ_s̄ => mean_fitness_dist[2],
        ),
        :multithread => true,
    )

    # Run inference in a multithread fasshion
    BayesFitness.mcmc.mcmc_mutant_fitness(; param...)

end # for