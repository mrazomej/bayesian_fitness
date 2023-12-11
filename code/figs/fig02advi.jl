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

# Import basic math
import StatsBase
import Distributions
import LinearAlgebra

# Import plotting libraries
using CairoMakie
import ColorSchemes
import ColorTypes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

# Import library to set random seed
import Random

Random.seed!(42)


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
       "output/advi_meanfield_01samples_3000steps.csv"

# Read ADVI results
df_advi = CSV.read(file, DF.DataFrame)

# Extract bc fitness values
df_fitness = df_advi[(df_advi.vartype.=="bc_fitness"), :]

# Extract real fitness values
df_fitness_truth = DF.rename!(
    unique(data[.!(data.neutral), [:barcode, :fitness]]),
    :barcode => :id
)

# Add real fitness to ADVI results
DF.leftjoin!(df_fitness, df_fitness_truth; on=:id)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples
n_samples = 10_000

# Sample from posterior MvNormal
df_samples = DF.DataFrame(
    Random.rand(
        Distributions.MvNormal(
            df_advi.mean, LinearAlgebra.Diagonal(df_advi.std .^ 2)
        ),
        n_samples
    )',
    df_advi.varname
)

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

# Import personal color palette
color_palette = BayesFitUtils.viz.colors()

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
bc_plot = StatsBase.sample(
    eachrow(DF.sort(df_fitness, :mean)),
    n_row * n_col,
    replace=false,
    ordered=true
)

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
                ColorSchemes.Purples_9, LinRange(0.5, 1.0, length(qs))
            )

            # Compute posterior predictive checks
            local ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
                df_samples, n_ppc; param=param
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
            data[data.barcode.==bc_plot[counter].id, :], :time
        )

        # Define colors
        local colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :bc_mean_fitness => Symbol(bc_plot[counter].varname),
            :bc_std_fitness => Symbol(
                replace(bc_plot[counter].varname, "s" => "logσ")
            ),
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        local ppc_mat = BarBay.stats.logfreq_ratio_bc_ppc(
            df_samples, n_ppc; param=param
        )
        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors
        )

        # Add scatter of data
        scatterlines!(ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.0)

        # Define fitness ranges to display in title
        vals = [
            round(bc_plot[counter].mean; sigdigits=2),
            round(bc_plot[counter].std; sigdigits=2),
        ]

        # Add title
        ax.title = "s⁽ᵐ⁾= $(vals[1])±$(vals[2])"

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

fig

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
    repeat([[minimum(df_fitness.mean), maximum(df_fitness.mean)]], 2)...;
    color=:black
)

# Add x-axis error bars
errorbars!(
    ax,
    df_fitness.fitness,
    df_fitness.mean,
    df_fitness.std,
    color=(:gray, 0.5),
    direction=:y
)

# Plot comparison
scatter!(ax, df_fitness.fitness, df_fitness.mean, markersize=8)

# ---------------------------------------------------------------------------- #

# Add axis
ax = Axis(
    gl_comp[2, 1],
    xlabel="ground truth fitness |z-score|",
    ylabel="ECDF",
    aspect=AxisAspect(1),
    xticks=LinearTicks(4),
    yticks=LinearTicks(4),
)

# Plot ECDF
ecdfplot!(
    ax,
    abs.((df_fitness.mean .- df_fitness.fitness) ./ df_fitness.std),
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
#