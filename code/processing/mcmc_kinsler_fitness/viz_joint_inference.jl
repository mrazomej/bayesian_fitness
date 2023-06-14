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
import PDFmerger

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
df = CSV.read("$(git_root())/data/kinsler_2020/tidy_counts.csv", DF.DataFrame)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Listing MCMC files and metadata
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# List Inference files available
mcmc_files = Glob.glob("./output/kinsler*.jld2")

# Initialize dataframe to save metadata
df_files = DF.DataFrame()

# Loop through files
for f in mcmc_files
    # Split and keep metadata variables
    vars = split(split(f, "/")[end], "_")
    # Extract metadata variables
    env = replace(vars[2], "env" => "")
    rep = replace(vars[3], "rep" => "")
    rm_T0 = parse(Bool, replace(vars[4], "rmT0.jld2" => ""))

    # Add info to dataframe
    DF.append!(
        df_files,
        DF.DataFrame(
            [[env], [rep], [rm_T0], [f]], ["env", "rep", "rm_T0", "file"]
        )
    )
end # for

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plotting PPC for population mean fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define dictionary with corresponding parameters for variables needed for the
# posterior predictive checks
param = Dict(
    :population_mean_fitness => :s̲ₜ,
    :population_std_fitness => :σ̲ₜ,
)

# Define number of posterior predictive check samples
n_ppc = 500

# Define quantiles to compute
qs = [0.68, 0.95, 0.997]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(qs)))

# Loop through files
for d in eachrow(df_files)
    # Select data
    data = df[(df.env.==d.env).&(df.rep.==d.rep), :]

    # Establish time for x-axis
    if d.rm_T0
        time = sort(unique(data.time))[3:end]
    else
        time = sort(unique(data.time))[2:end]
    end # if

    # Load chain
    chn = JLD2.load(d.file)["chain"]

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
        chn, n_ppc; param=param
    )

    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="PPC $(d.env) | $(d.rep)"
    )

    # Plot posterior predictive checks
    BayesFitness.viz.ppc_time_series!(
        ax, qs, ppc_mat; time=time, colors=colors
    )

    # Add plot for median (we use the 5 percentile to have a "thicker" line showing
    # the median)
    BayesFitness.viz.ppc_time_series!(
        ax, [0.05], ppc_mat; time=time, colors=ColorSchemes.Blues_9[end:end]
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
    save("./output/figs/temp.pdf", fig)

    # Append pdf
    PDFmerger.append_pdf!(
        "./output/figs/logfreqratio_ppc_neutral.pdf",
        "./output/figs/temp.pdf",
        cleanup=true
    )
end # for

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot PPC for a few mutants
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of plots
n_plots_row = 4
n_plots_col = 4

# Define number of posterior predictive check samples
n_ppc = 500

# Define quantiles to compute
qs = [0.68, 0.95, 0.997]
# Define quantile for plot title
q = 0.68

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 0.75, length(qs)))

# Loop through files
# for d in eachrow(df_files)
# Select data
data = df[(df.env.==d.env).&(df.rep.==d.rep), :]

# Establish time for x-axis
if d.rm_T0
    time = sort(unique(data.time))[3:end]
else
    time = sort(unique(data.time))[2:end]
end # if

# Compute naive fitness estimate
df_fit = BayesFitness.stats.naive_fitness(data; rm_T0=d.rm_T0)
# Sort by fitness
DF.sort!(df_fit, :fitness)

# Select evenly-distributed barcodes to plot
bcs = df_fit.barcode[
    1:size(df_fit, 1)÷(n_plots_col*n_plots_row):end
][1:(n_plots_col*n_plots_row)]

# Load chain and mutants order
chn, mutnames = values(JLD2.load(d.file))

# Initialize figure
fig = Figure(resolution=(700, 700))

# Add grid layout to have better manipulation
gl = fig[1, 1:4] = GridLayout()

# Add axis
axes = [
    Axis(
        gl[i, j],
    ) for i = 1:n_plots_row for j = 1:n_plots_col
]

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=20)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)
# Add Plot title
Label(fig[0, 2:3], text="PPC $(d.env) | $(d.rep)", fontsize=20)
# Set row and col gaps
colgap!(gl, 10)
rowgap!(gl, 10)


# # Loop through barcodes
for (i, bc) in enumerate(bcs)
    # Define dictionary with corresponding parameters for variables needed
    # for the posterior predictive checks
    param = Dict(
        :population_mean_fitness => :s̲ₜ,
        :mutant_mean_fitness => Symbol("s̲⁽ᵐ⁾[$(findfirst(mutnames.==bc))]"),
        :mutant_std_fitness => Symbol("σ̲⁽ᵐ⁾[$(findfirst(mutnames.==bc))]")
    )
    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
        chn, n_ppc; param=param
    )

    # Plot posterior predictive checks
    # BayesFitness.viz.ppc_time_series!(
    #     axes[i], qs, ppc_mat; time=time, colors=colors
    # )

    # Add plot for median (we use the 5 percentile to have a "thicker" line
    # showing the median)
    # BayesFitness.viz.ppc_time_series!(
    #     axes[i],
    #     [0.05],
    #     ppc_mat;
    #     time=time,
    #     colors=ColorSchemes.Blues_9[end:end]
    # )

    # Plot log-frequency ratio of neutrals
    BayesFitness.viz.logfreq_ratio_time_series!(
        axes[i],
        data[data.barcode.==bc, :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2,
        markersize=10
    )

    # Compute mutant fitness median for title
    s_median = round(
        StatsBase.median(chn[param[:mutant_mean_fitness]]), sigdigits=2
    )
    s_min = round(
        StatsBase.percentile(
            chn[param[:mutant_mean_fitness]], 
            [(1.0 - (q / 2)]
        ), 
        sigdigits=2
    )

    # Compute upper and lower percentile

    # Add title
    axes[i].title = "BC $(bc)|s⁽ᵐ⁾=$(s⁽ᵐ⁾)"
    axes[i].titlesize = 11

    # Hide axis decorations
    hidedecorations!(axes[i], grid=false)
end # for

fig

#     # Save figure into pdf
#     save("./output/figs/temp.pdf", fig)

#     # Append pdf
#     PDFmerger.append_pdf!(
#         "./output/figs/logfreqratio_ppc_mutant.pdf",
#         "./output/figs/temp.pdf",
#         cleanup=true
#     )

# end # for

