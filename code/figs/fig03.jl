##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

import Revise
# Import project package
import BayesFitUtils

# Import library to perform Bayesian inference
import BarBay

# Import basic math
import Random
import Distributions
import LinearAlgebra
import StatsBase

# Import libraries to manipulate df_counts
import DataFrames as DF
import CSV

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes
import ColorTypes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

Random.seed!(42)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import df_counts
df_counts = CSV.read(
    "$(git_root())/data/logistic_growth/data_003/tidy_data.csv",
    DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load ADVI results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = "$(git_root())/code/processing/" *
       "data003_logistic_1000bc_03env_01rep/" *
       "output/advi_meanfield_01samples_10000steps.csv"

# Convert results to tidy DataFrame
df_advi = CSV.read(file, DF.DataFrame)

# Extract bc fitness values
df_fitness = df_advi[(df_advi.vartype.=="bc_fitness"), :]

# Extract and append ground truth fitness values
DF.leftjoin!(
    df_fitness,
    DF.rename(
        unique(df_counts[.!(df_counts.neutral), [:barcode, :fitness, :env]]),
        :barcode => :id
    );
    on=[:id, :env]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into DataFrame
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
fig = Figure(resolution=(1200, 750))

# Add grid layout for illustrator diagram
gl_illustrator = fig[1, 1] = GridLayout()

# Add grid lauout for df_countssets
gl_data = fig[2, 1] = GridLayout()

# Add grid layout for true hyperfitness vs true fitness
gl_true = fig[1, 2:4] = GridLayout()

# Add grid layout for ECDF plots
gl_ecdf = fig[2, 2] = GridLayout()

# Add grid layout for posterior predictive checks
gl_ppc = fig[2, 3:4] = GridLayout()

# Define unique environments
env_unique = sort(unique(df_counts.env))

# Define color for environments to keep consistency
env_colors = Dict(env_unique .=> ColorSchemes.tableau_10[5:7])

# ----------------------------------------------------------------------------

# Add axis
ax = Axis(
    gl_data[1, 1],
    xlabel="time [dilution cycles]",
    ylabel="ln(fₜ₊₁/fₜ)",
    aspect=AxisAspect(1),
    backgroundcolor=:white
)

# Define time-environment relation
time_env = Matrix(unique(df_counts[:, [:time, :env]]))

# Loop through each time point
for t = 2:size(time_env, 1)
    # Color plot background
    vspan!(
        ax,
        time_env[t, 1] - 0.5,
        time_env[t, 1] + 0.5,
        color=(env_colors[time_env[t, 2]], 0.25)
    )
end # for

# Plot log-frequency ratio of neutrals
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    df_counts[.!df_counts.neutral, :];
    freq_col=:freq,
    alpha=0.15,
    linewidth=2
)

# Plot log-frequency ratio of neutrals
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    df_counts[df_counts.neutral, :];
    freq_col=:freq,
    linewidth=2,
    color=ColorSchemes.Blues_9[end]
)

# ----------------------------------------------------------------------------

# Add axis
ax = [
    Axis(
        gl_true[1, i],
        xlabel="ground truth fitness",
        ylabel="inferred fitness",
        aspect=AxisAspect(1),
    ) for i = 1:3
]

# Group data by environment
df_group = DF.groupby(df_fitness, :env)

# Loop through environments
for (i, data) in enumerate(df_group)
    # Add identity line
    lines!(
        ax[i],
        repeat(
            [[minimum(data.fitness) * 1.1, maximum(data.fitness) * 1.1]],
            2
        )...,
        linestyle=:dash,
        color="black",
    )

    # Error bars
    errorbars!(
        ax[i],
        data.fitness,
        data.mean,
        data.std,
        color=(:gray, 0.5),
        direction=:y,
    )

    # Add points
    scatter!(
        ax[i],
        data.fitness,
        data.mean,
        markersize=7,
        color=(env_colors[i], 0.5)
    )

    ax[i].title = "environment $(i)"
end # for

# ----------------------------------------------------------------------------

# Add axis for fitness ECDF
ax = Axis(
    gl_ecdf[1, 1],
    xlabel="ground truth fitness |z-score|",
    ylabel="ECDF",
    aspect=AxisAspect(1)
)

# Plot ECDF for all environments
ecdfplot!(
    ax,
    abs.((df_fitness.mean .- df_fitness.fitness) ./ df_fitness.std),
    color=:black,
    label="all",
    linewidth=2.5,
)

# Loop through environments
for (i, data) in enumerate(df_group)
    # Plot ECDF for single enviroment
    ecdfplot!(
        ax,
        abs.((data.mean .- data.fitness) ./ data.std),
        color=env_colors[i],
        label="env $(i)",
        linewidth=2.5,
    )
end # for

# Add legend
axislegend(ax, position=:rb, framevisible=false)

# ---------------------------------------------------------------------------- #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [2, 3]

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
        local ax = Axis(
            gl_ppc[row, col], aspect=AxisAspect(1.5), backgroundcolor=:white
        )

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
                df_samples, n_ppc; model=:normal, param=param
            )

            # Define time-environment relation
            local time_env = Matrix(unique(df_counts[:, [:time, :env]]))

            # Define time
            local t = vec(collect(axes(ppc_mat, 2)) .+ 1)

            # Loop through each time point
            for t = 2:size(time_env, 1)
                # Color plot background
                vspan!(
                    ax,
                    time_env[t, 1] - 0.5,
                    time_env[t, 1] + 0.5,
                    color=(env_colors[time_env[t, 2]], 0.25)
                )
            end # for

            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax, qs, ppc_mat; colors=colors, time=t
            )

            # Plot log-frequency ratio of neutrals
            BayesFitUtils.viz.logfreq_ratio_time_series!(
                ax,
                df_counts[df_counts.neutral, :];
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

        # Define colors
        local colors = get(
            ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs))
        )

        # Extract data
        data_bc = DF.sort(
            df_counts[df_counts.barcode.==bc_plot[counter].id, :], :time
        )
        # Extract fitness variable names
        s_var = df_fitness[(df_fitness.id.==bc_plot[counter].id), :varname]
        # Extract logσ variable names
        σ_var = replace.(s_var, Ref("s" => "logσ"))
        # Extract mean fitness variables
        sₜ_var = df_advi[(df_advi.vartype.=="pop_mean_fitness"), :varname]

        # Extract samples
        global df_bc = df_samples[:, [sₜ_var; s_var; σ_var]]

        # Define colors
        local ppc_color = get(
            ColorSchemes.Purples_9, LinRange(0.5, 1.0, length(qs))
        )

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :bc_mean_fitness => :s̲⁽ᵐ⁾,
            :bc_std_fitness => :logσ̲⁽ᵐ⁾,
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        local ppc_mat = BarBay.stats.logfreq_ratio_multienv_ppc(
            df_bc, n_ppc, data_bc.env; model=:normal, param=param
        )

        # Define time-environment relation
        local time_env = Matrix(unique(data_bc[:, [:time, :env]]))

        # Define time
        local t = vec(collect(axes(ppc_mat, 2)) .+ 1)

        # Loop through each time point
        for t = 2:size(time_env, 1)
            # Color plot background
            vspan!(
                ax,
                time_env[t, 1] - 0.5,
                time_env[t, 1] + 0.5,
                color=(env_colors[time_env[t, 2]], 0.25)
            )
        end # for

        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors, time=t
        )

        # Add scatter of data
        scatterlines!(
            ax, t, diff(log.(data_bc.freq)), color=:black, linewidth=2.0
        )

        # Define fitness ranges to display in title
        vals = [
            round.(
                df_fitness[df_fitness.id.==bc_plot[counter].id, :mean];
                sigdigits=2
            ),
            round.(
                df_fitness[df_fitness.id.==bc_plot[counter].id, :std];
                sigdigits=2
            ),
        ]

        # Add title
        ax.title = "s₁⁽ᵐ⁾=$(vals[1][1])±$(vals[2][1])\n" *
                   "s₂⁽ᵐ⁾=$(vals[1][2])±$(vals[2][2])\n" *
                   "s₃⁽ᵐ⁾=$(vals[1][3])±$(vals[2][3])"
        ax.titlesize = 14

        ## == Plot format == ##

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(gl_ppc[end, :, Bottom()], "time [dilution cycles]", fontsize=18)
# Add y-axis label
Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=18)
# # Set spacing
rowgap!(gl_ppc, 0)
colgap!(gl_ppc, 0)

# ---------------------------------------------------------------------------- #

# Add subplot labels
Label(
    gl_illustrator[1, 1, TopLeft()], "(A)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_data[1, 1, TopLeft()], "(B)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_true[1, 1, TopLeft()], "(C)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_true[1, 2, TopLeft()], "(D)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_true[1, 3, TopLeft()], "(E)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_ecdf[1, 1, TopLeft()], "(F)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_ppc[1, 1, TopLeft()], "(G)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

save("$(git_root())/doc/figs/fig03B-G.pdf", fig)

fig

