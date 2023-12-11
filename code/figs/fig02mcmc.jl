##
println("Loading packages...")



import Revise
# Import project package
import BayesFitUtils
# Import package for Bayesian inference
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

# Import basic math
import StatsBase

# Import library to read MCMC chains
import JLD2

# Import plotting libraries
using CairoMakie
import ColorSchemes
import ColorTypes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.pboc_makie!()

# Import library to set random seed
import Random

Random.seed!(42)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_001/tidy_data.csv", DF.DataFrame
)

# Define file
file = "$(git_root())/code/processing/data001_logistic_250bc_01env_01rep/" *
       "output/chain_joint_fitness_1000steps_04walkers.jld2"

# Load chain
ids, chn = values(JLD2.load(file))
# Remove the string "mut" from mutant names
mut_num = replace.(ids, "mut" => "")

# Find columns with mutant fitness values and error
s_names = MCMCChains.namesingroup(chn, :s̲⁽ᵐ⁾)
σ_names = MCMCChains.namesingroup(chn, :σ̲⁽ᵐ⁾)

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

# Add barcode column to match
df_filt[!, :barcode] = getindex.(
    Ref(Dict(unique(df_filt.variable) .=> ids)), df_filt.variable
)

# Append fitness and growth rate value
DF.leftjoin!(
    df_filt,
    unique(data[.!(data.neutral), [:barcode, :fitness, :growth_rate]]);
    on=:barcode
)

# Compute summary statistics
df_summary = DF.combine(
    DF.groupby(df_filt, [:variable, :barcode]),
    :value => StatsBase.median,
    :value => StatsBase.mean,
    :value => StatsBase.std,
    :value => StatsBase.var,
    :value => StatsBase.skewness,
    :value => StatsBase.kurtosis,
)

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
# Set figure layout
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(1200, 600))

# Add grid layout for datasets
gl_data = fig[1, 1] = GridLayout()

# Add grid layout for posterior predictive checks
gl_ppc = fig[1, 2:3] = GridLayout()

# Add grid layout for comparison
gl_comp = fig[1, 4] = GridLayout()

# ---------------------------------------------------------------------------- #

# Add axis
ax = [
    Axis(
        gl_data[i, 1],
        xlabel="time [dilution cycles]",
        aspect=AxisAspect(1)
    ) for i = 1:2
]

# Plot mutant barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax[1],
    data[.!(data.neutral), :],
    quant_col=:freq,
    zero_lim=0,
    alpha=0.3
)

# Plot neutral barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax[1],
    data[data.neutral, :],
    quant_col=:freq,
    zero_lim=0,
    color=ColorSchemes.Blues_9[end],
)

# Change scale
ax[1].yscale = log10
# Add label
ax[1].ylabel = "barcode frequency"

# Plot mutant barcode trajectories
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax[2],
    data[.!(data.neutral), :],
    alpha=0.3
)

# Plot neutral barcode trajectories
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax[2],
    data[data.neutral, :],
    color=ColorSchemes.Blues_9[end],
)

# Add label
ax[2].ylabel = "ln(fₜ₊₁/fₜ)"

# Define gap between subplots
rowgap!(gl_data, 0)

# ---------------------------------------------------------------------------- #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [3, 3]

# List example barcodes to plot
bc_plot = StatsBase.sample(eachrow(df_summary), n_row * n_col)

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add axis
        local ax = Axis(gl_ppc[row, col], aspect=AxisAspect(1.25))

        # Check if first first entry
        if (row == 1) & (col == 1)
            # Define dictionary with corresponding parameters for variables
            # needed for the posterior predictive checks
            param = Dict(
                :population_mean_fitness => :s̲ₜ,
                :population_std_fitness => :σ̲ₜ,
            )

            # Define colors
            local colors = get(
                ColorSchemes.Purples_9, LinRange(0.25, 1.0, length(qs))
            )

            # Compute posterior predictive checks
            local ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
                chn, n_ppc; param=param
            )

            # Define time
            t = vec(collect(axes(ppc_mat, 2)) .+ 1)

            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax, qs, ppc_mat; colors=colors, time=t
            )

            # Plot log-frequency ratio of neutrals
            BayesFitUtils.viz.logfreq_ratio_time_series!(
                ax,
                data[data.neutral, :];
                freq_col=:freq,
                color=:black,
                alpha=1.0,
                linewidth=1.5
            )

            # Hide axis decorations
            hidedecorations!.(ax, grid=false)

            ax.title = "neutral lineages"
            # ax.titlesize = 18

            counter += 1

            continue
        end # if

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
        local ppc_mat = BarBay.stats.logfreq_ratio_mutant_ppc(
            chn, n_ppc; param=param
        )
        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors
        )

        # Add scatter of data
        scatterlines!(ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.0)

        # Define fitness ranges to display in title
        vals = [
            round(bc_plot[counter].median; sigdigits=2),
            round(
                abs(
                    bc_plot[counter]["16.0_percentile"] -
                    bc_plot[counter].median
                ),
                sigdigits=2
            ),
            round(
                abs(
                    bc_plot[counter]["84.0_percentile"] -
                    bc_plot[counter].median
                ),
                sigdigits=2
            )
        ]

        # Add title
        ax.title = L"s^{(m)} = %$(vals[1])_{-%$(vals[2])}^{+%$(vals[3])}"
        # ax.titlesize = 18

        ## == Plot format == ##

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(gl_ppc[end, :, Bottom()], "time [dilution cycles]", fontsize=22)
# Add y-axis label
Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=22)
# Set spacing
rowgap!(gl_ppc, 0)
colgap!(gl_ppc, 4)

# ---------------------------------------------------------------------------- #

# Add axis
ax = Axis(
    gl_comp[1, 1],
    xlabel="ground truth fitness",
    ylabel="inferred fitness",
    aspect=AxisAspect(1)
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

# ---------------------------------------------------------------------------- #

# Group data by barcode
df_group = DF.groupby(df_filt, :variable)

# Initialize matrix to save difference between ground truth and inferred value
diff_mat = Matrix{Float64}(undef, size(first(df_group), 1), length(df_group))

# Loop through mutants
for (i, var) in enumerate(df_group)
    # Extract true value
    fitness = df_summary[df_summary.variable.==first(var.variable), :fitness]
    # Compute differences
    diff_mat[:, i] = abs.(var.value .- fitness)
end # for

# Add axis
ax = Axis(
    gl_comp[2, 1],
    xlabel="|median - true value|",
    ylabel="ECDF",
    aspect=AxisAspect(1),
    xticks=LinearTicks(4),
    yticks=LinearTicks(4),
)

# Group data by barcode
df_group = DF.groupby(df_filt, [:variable, :barcode])

# # Initialize matrix to save difference between ground truth and inferred value
diff_mat = Matrix{Float64}(undef, size(first(df_group), 1), length(df_group))

# Loop through mutants
for (i, var) in enumerate(df_group)
    # Compute differences
    diff_mat[:, i] = var.value .- var.fitness
end # for

# Add reference line
vlines!(ax, 0, linestyle=:dash, color=:black)

# Plot ECDF
ecdfplot!(
    ax,
    abs.(StatsBase.percentile.(eachcol(diff_mat), Ref(50.0))),
    linewidth=2
)

# ---------------------------------------------------------------------------- #

# Add subplot labels
Label(
    gl_data[1, 1, TopLeft()], "(A)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

Label(
    gl_data[2, 1, TopLeft()], "(B)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

Label(
    gl_ppc[1, 1, TopLeft()], "(C)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

Label(
    gl_comp[1, 1, TopLeft()], "(D)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

Label(
    gl_comp[2, 1, TopLeft()], "(E)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

save("$(git_root())/doc/figs/fig02.pdf", fig)
save("$(git_root())/doc/figs/fig02.png", fig)

fig
##