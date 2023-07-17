##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import project package
import BayesFitUtils

# Import library package
import BayesFitness

# Import basic math
import StatsBase

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import library to list files
import Glob

# Import library to load files
import JLD2

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.pboc_makie!()

import Random

Random.seed!(42)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

# Generate figure dictionary
if !isdir("./output/figs/")
    mkdir("./output/figs/")
end # if

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_001/tidy_data.csv", DF.DataFrame
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plotting PPC for population mean fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = first(Glob.glob("./output/chain_popmean_fitness_*"))

# Check if file exists
if isfile(file)

    # Define dictionary with corresponding parameters for variables needed for the
    # posterior predictive checks
    param = Dict(
        :population_mean_fitness => :s̲ₜ,
        :population_std_fitness => :σ̲ₜ,
    )

    # Define number of posterior predictive check samples
    n_ppc = 500

    # Define quantiles to compute
    qs = [0.05, 0.68, 0.95]

    # Define colors
    colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1.0, length(qs)))

    # Remove old version of file
    rm("./output/figs/logfreqratio_ppc_neutral.pdf", force=true)

    # Load chain
    chn = JLD2.load(file)["chain"]

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
        chn, n_ppc; param=param
    )

    # Define time
    t = vec(collect(axes(ppc_mat, 2)) .+ 1)

    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="neutral lineages PPC"
    )

    # Plot posterior predictive checks
    BayesFitness.viz.ppc_time_series!(
        ax, qs, ppc_mat; colors=colors, time=t
    )

    # Plot log-frequency ratio of neutrals
    BayesFitness.viz.logfreq_ratio_time_series!(
        ax,
        data[data.neutral, :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2
    )

    # Save figure into pdf
    save("./output/figs/logfreqratio_ppc_neutral.pdf", fig)

end # if