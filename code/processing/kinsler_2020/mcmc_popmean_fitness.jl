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
import PDFmerger

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

# Define output directory
outdir = "./output/popmean_fitness"

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

if !isdir(outdir)
    mkdir(outdir)
end # if

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading the data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
df = CSV.read("$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame)

# Read datasets visual evaluation info
df_include = CSV.read(
    "$(git_root())/data/kinsler_2020/exp_include.csv", DF.DataFrame
)

# NOTE: For the multithread to run when some files have already been processed,
# we need to avoid the "continue" condition when checking if a file was already
# processed (I honestly don't know why). Therefore, we must list all datasets to
# be processed, see which ones are complete, and remove them from the data fed
# to the sampler. This allows Threads.@threads to actually work.

# Initialize array indicating if dataset has been completely processed or not
complete_bool = repeat([false], size(df_include, 1))

# Loop through groups
for i in axes(df_include, 1)
    # Extract info
    env, rep, rm_T0 = collect(df_include[i, :])

    # Define output directory
    fname = "$(outdir)/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0.jld2"

    # Check if output doesn't exist
    if isfile(fname)
        # Idicate file is not complete
        complete_bool[i] = true
    end # if
end # for

println("Number of datasets previously processed: $(sum(complete_bool))")

# filtered datasets to process
df_include = df_include[.!(complete_bool), :]

println("$(size(df_include, 1)) datasets to process...\n")

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Initialize MCMC sampling
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Initializing MCMC sampling...\n")

# Loop through datasets
for i in axes(df_include, 1)
    # Extract info
    env, rep, rm_T0 = collect(df_include[i, :])

    println("Processing $(env) | $(rep)")

    # Extract data
    data = df[(df.env.==env).&(df.rep.==rep), :]

    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => n_walkers,
        :n_steps => n_steps,
        :outputname => "$(outdir)/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0_$(n_steps)steps_$(lpad(2, n_walkers, "0")walkers)",
        :model => BarBay.model.neutrals_lognormal,
        :sampler => Turing.DynamicNUTS(),
        :ensemble => Turing.MCMCThreads(),
        :rm_T0 => rm_T0,
    )

    # Run inference
    println("Running Inference for group $(i)...")

    try
        @time BarBay.mcmc.mcmc_popmean_fitness(; param...)
    catch
        @warn "Group $(i) was already processed"
    end
end # for

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate diagnostic plots
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

if gen_plots
    # List all output directories
    out_dirs = Glob.glob("./output/*_R*")

    # Loop through directories
    for (i, od) in enumerate(out_dirs)
        if i % 10 == 0
            println("Generating plot #$(i) out of $(length(out_dirs))")
        end # if

        # Concatenate population mean fitness chains into single chain
        chains = BarBay.utils.var_jld2_concat(od, "meanfitness", :sₜ)

        # Initialize figure
        fig = Figure(resolution=(800, 800))

        # Generate mcmc_trace_density! plot
        BayesFitUtils.viz.mcmc_trace_density!(
            fig, chains; alpha=0.5, title=String(split(od, "/")[end])
        )

        # Save figure into pdf
        save("./output/temp.pdf", fig)

        # Append pdf
        PDFmerger.append_pdf!(
            "./output/meanfitness_trace_density.pdf",
            "./output/temp.pdf",
            cleanup=true
        )
    end # for

    println("Done printing population mean fitness trace/density plots!")
end # if