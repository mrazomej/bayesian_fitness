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
import Distributions
import Random

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

# Define dictionary with corresponding parameters for variables needed for
# the posterior predictive checks
param = Dict(
    :population_mean_fitness => :s̲ₜ,
    :population_std_fitness => :σ̲ₜ,
)

# Define number of posterior predictive check samples
n_ppc = 500

# Define quantiles to compute
qs = [0.05, 0.68, 0.95]

# Define colors
colors = get(ColorSchemes.Purples_9, LinRange(0.25, 1.0, length(qs)))

# Remove old version of file
rm("./output/figs/logfreqratio_ppc_neutral_prior.pdf", force=true)

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
    title="neutral lineages PPC | Prior"
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
save("./output/figs/logfreqratio_ppc_neutral_prior.pdf", fig)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read chain into memory and generate tidy dataframe with it
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = first(Glob.glob("./output/chain_joint_fitness_*"))

# Load chain
ids, chn = values(JLD2.load(file))
# Remove the string "mut" from mutant names
mut_num = replace.(ids, "mut" => "")

# Find columns with mutant fitness values and error
s_names = MCMCChains.namesingroup(chn, :s̲⁽ᵐ⁾)
σ_names = MCMCChains.namesingroup(chn, :σ̲⁽ᵐ⁾)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for neutral lineages in joint inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# # Define dictionary with corresponding parameters for variables needed for
# the posterior predictive checks
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
rm("./output/figs/logfreqratio_ppc_neutral_posterior.pdf", force=true)

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
    title="neutral lineages PPC | Posterior"
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
save("./output/figs/logfreqratio_ppc_neutral_posterior.pdf", fig)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute summary statistics for fitness values
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define percentiles to include
per = [2.5, 97.5, 16, 84]

# Convert chain to tidy dataframe
df = DF.stack(DF.DataFrame(chn[s_names]), DF.Not([:iteration, :chain]))

# FILTER 95% #

# Group data by barcode
df_group = DF.groupby(df, :variable)

# Initialize dataframe to save summaries
df_filt = DF.DataFrame()

# Loop through groups
for g in df_group
    # Filter data
    DF.append!(
        df_filt,
        g[(g.value.≥StatsBase.percentile(g.value, 5)).&(g.value.≤StatsBase.percentile(g.value, 95)),
            :]
    )
end # group

# Compute summary statistics
df_summary = DF.combine(
    DF.groupby(df_filt, :variable),
    :value => StatsBase.median,
    :value => StatsBase.mean,
    :value => StatsBase.std,
    :value => StatsBase.var,
    :value => StatsBase.skewness,
    :value => StatsBase.kurtosis,
)

# Add barcode column to match
df_summary[!, :barcode] = [
    Dict(unique(df_summary.variable) .=> ids)[x] for x in df_summary.variable
]

# Loop through percentiles
for p in per
    # Compute and add percentile
    DF.leftjoin!(
        df_summary,
        # Rename column from :value_function to :p_percentile
        DF.rename!(
            # Compute percentile p for each group
            DF.combine(
                # Group MCMC chains by :variable, :value
                DF.groupby(df_filt[:, [:variable, :value]], :variable),
                # Define anonymous function to compute percentile
                :value => x -> StatsBase.percentile(x, p)
            ),
            :value_function => Symbol("$(p)_percentile")
        );
        on=:variable
    )
end # for

# Rename columns
DF.rename!(
    df_summary,
    :value_median => :median,
    :value_mean => :mean,
    :value_std => :std,
    :value_var => :var,
    :value_skewness => :skewness,
    :value_kurtosis => :excess_kurtosis,
)

# Append fitness and growth rate value
DF.leftjoin!(
    df_summary,
    unique(data[.!(data.neutral), [:barcode, :fitness, :growth_rate]]);
    on=:barcode
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 400))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="true fitness value",
    ylabel="inferred fitness",
)

# Plot identity line
lines!(
    ax,
    repeat([[minimum(df_summary.median), maximum(df_summary.median)]], 2)...;
    color=:black
)

# Add x-axis error bars
errorbars!(
    ax,
    df_summary.fitness,
    df_summary.median,
    abs.(df_summary.median .- df_summary[:, "16.0_percentile"]),
    abs.(df_summary.median .- df_summary[:, "84.0_percentile"]),
    color=(:gray, 0.5),
    direction=:y
)

# Plot comparison
scatter!(ax, df_summary.fitness, df_summary.median, markersize=8)

save("./output/figs/fitness_comparison.pdf", fig)

fig
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(df_summary.median .- df_summary.fitness))

save("./output/figs/median_true_ecdf.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Fit Gaussian distribution to chains
dist_fit = [Distributions.fit(Distributions.Normal, chn[x]) for x in s_names]
# Extract parameters
dist_param = hcat(
    first.(Distributions.params.(dist_fit)),
    last.(Distributions.params.(dist_fit))
)
# Compute Z-score of true fitness values
fitness_zscore = (df_summary.fitness .- dist_param[:, 1]) ./ dist_param[:, 2]
# Initialize figure
fig = Figure(resolution=(400, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|z-score|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(fitness_zscore))

save("./output/figs/zscore_ecdf.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot example posterior predictive checks
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [4, 4]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# List example barcodes to plot
bc_plot = StatsBase.sample(eachrow(df_summary), n_row * n_col)

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add axis
        local ax = Axis(fig[row, col])

        # Extract data
        data_bc = DF.sort(
            data[data.barcode.==bc_plot[counter].barcode, :], :time
        )

        # Define colors
        local colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :mutant_mean_fitness => Symbol(bc_plot[counter].variable),
            :mutant_std_fitness => Symbol(
                replace(bc_plot[counter].variable, "s" => "σ")
            ),
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        local ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
            chn, n_ppc; param=param
        )
        # Plot posterior predictive checks
        BayesFitness.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors
        )

        # Add scatter of data
        scatterlines!(ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.5)

        # Add title
        ax.title = "barcode $(first(data_bc.barcode))"
        ax.titlesize = 18

        ## == Plot format == ##

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=22)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=22)

fig

save("./output/figs/logfreqratio_ppc_mutant.pdf", fig)