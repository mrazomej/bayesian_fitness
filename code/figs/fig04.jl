##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

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

println("Loading simulated dataset...")
# Import data
df_counts = CSV.read(
    "$(git_root())/data/logistic_growth/data_002/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load ADVI results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results...")

# Define file
file = "$(git_root())/code/processing/data002_logistic_1000bc_01env_02rep/" *
       "output/advi_meanfield_01samples_5000steps.csv"

# Convert results to tidy dataframe
df_advi = CSV.read(file, DF.DataFrame)

# Split variables by replicate
rep_vars = Dict(
    Symbol(rep) => df_advi[df_advi.rep.==rep, :varname]
    for rep in unique(df_advi.rep)
)
# Remove "N/A" from dictionary
delete!(rep_vars, Symbol("N/A"))

# Extract bc fitness values
df_fitness = df_advi[(df_advi.vartype.=="bc_fitness"), :]

# Extract and append hyperfitness values for each fitness value
DF.leftjoin!(
    df_fitness,
    DF.rename(
        df_advi[(df_advi.vartype.=="bc_hyperfitness"),
            [:mean, :std, :varname, :id]],
        :mean => :mean_h,
        :std => :std_h,
        :varname => :varname_h
    );
    on=:id
)

# Extract and append ground truth fitness and hyperfitness values
DF.leftjoin!(
    df_fitness,
    DF.rename(
        unique(
            df_counts[.!(df_counts.neutral),
                [:barcode, :fitness, :hyperfitness, :rep]]
        ),
        :barcode => :id
    );
    on=[:id, :rep]
)

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
fig = Figure(resolution=(1200, 1000))

# Add global GridLayout
gl_fig = fig[1, 1] = GridLayout()

# Add grid layout for illustrator diagram
gl_illustrator = gl_fig[1, 1] = GridLayout()

# Add grid lauout for datasets
gl_data = gl_fig[1, 2:3] = GridLayout()

# Add grid layout for true hyperfitness vs true fitness
gl_hyper_vs_fit_true = gl_fig[1, 4] = GridLayout()

# Add grid layout for posterior predictive checks
gl_ppc = gl_fig[2:3, 1:2] = GridLayout()

# Add grid layout for inference vs true
gl_comp = gl_fig[2, 3:4] = GridLayout()

# Add grid layout for ECDF plots
gl_ecdf = gl_fig[3, 3:4] = GridLayout()

# Select replicate color
rep_color = ColorSchemes.seaborn_colorblind[1:length(unique(df_counts.rep))]
# ---------------------------------------------------------------------------- #

# Add axis
ax = [
    Axis(
        gl_data[1, i],
        xlabel="time [dilution cycles]",
        aspect=AxisAspect(1)
    ) for i = 1:2
]

# Group data by replicate
df_group = DF.groupby(df_counts, :rep)

# Loop through replicates
for (i, data) in enumerate(df_group)
    # Plot mutant barcode trajectories
    BayesFitUtils.viz.bc_time_series!(
        ax[i],
        data[.!(data.neutral), :],
        quant_col=:freq,
        zero_lim=1E-10,
        alpha=0.3,
        zero_label="extinct"
    )

    # Plot neutral barcode trajectories
    BayesFitUtils.viz.bc_time_series!(
        ax[i],
        data[data.neutral, :],
        quant_col=:freq,
        zero_lim=1E-10,
        color=ColorSchemes.Blues_9[end],
    )

    # Change scale
    ax[i].yscale = log10
    # Add label
    ax[i].ylabel = "barcode frequency"
    # Add title
    ax[i].title = "replicate $(first(data.rep))"
end # for

# ---------------------------------------------------------------------------- #

# Add axis for true hyperfiness vs true fitness
ax = Axis(
    gl_hyper_vs_fit_true[1, 1],
    xlabel="ground truth hyper-fitness",
    ylabel="ground truth replicate fitness",
    aspect=AxisAspect(1)
)

# Group data by replicate
df_group = DF.groupby(df_fitness, :rep)

# Loop through replicates
for (i, data) in enumerate(df_group)
    # Extract replicate
    rep = first(data.rep)
    # Plot true hyperfitness vs fitness
    scatter!(ax, data.hyperfitness, data.fitness, markersize=8, label="$(rep)")
end # for

# Add legend
axislegend(ax, position=:rb)

# ---------------------------------------------------------------------------- #

# Add axis for true vs inferred hyperfitness 
ax = Axis(
    gl_comp[1, 1],
    xlabel="ground truth hyper-fitness",
    ylabel="inferred hyper-fitness",
    aspect=AxisAspect(1)
)

# Select data
data = df_fitness[(df_fitness.rep.=="R1"), :]

# Add identity line
lines!(
    ax,
    repeat([[minimum(data.mean_h), maximum(data.mean_h)] .* 1.05], 2)...,
    linestyle=:dash,
    color=:black
)

# Plot errorbars
errorbars!(
    ax,
    data.hyperfitness,
    data.mean_h,
    data.std_h,
    color=(:gray, 0.5)
)

# Plot mean point
scatter!(
    ax,
    data.hyperfitness,
    data.mean_h,
    markersize=5,
    color=(ColorSchemes.seaborn_colorblind[3], 0.75)
)

# Add axis for true vs inferred fitness 
ax = Axis(
    gl_comp[1, 2],
    xlabel="ground truth replicate fitness",
    ylabel="inferred replicate fitness",
    aspect=AxisAspect(1)
)

# Add identity line
lines!(
    ax,
    repeat([[minimum(data.mean_h), maximum(data.mean_h)] .* 1.05], 2)...,
    linestyle=:dash,
    color=:black
)

# Group data
df_group = DF.groupby(df_fitness, :rep)

# Loop through replicates
for (i, data) in enumerate(df_group)
    # Plot errorbars
    errorbars!(
        ax,
        data.fitness,
        data.mean,
        data.std,
        color=(:gray, 0.5)
    )

end # for

# Loop through replicates
for (i, data) in enumerate(df_group)
    # Extract replicate
    rep = first(data.rep)
    # Plot mean point
    scatter!(
        ax,
        data.fitness,
        data.mean,
        markersize=5,
        label="$(rep)",
        color=(rep_color[i], 0.75)
    )
end # for

# Add legend
axislegend(ax, position=:rb)

# ---------------------------------------------------------------------------- #

# Add axis for hyperfitness ECDF
ax = Axis(
    gl_ecdf[1, 1],
    xlabel="ground truth hyper-fitness |z-score|",
    ylabel="ECDF",
    aspect=AxisAspect(1)
)
# Select data
data = df_fitness[(df_fitness.rep.=="R1"), :]

# Plot hyperfitness ECDF
ecdfplot!(
    ax,
    abs.((data.mean_h .- data.hyperfitness) ./ data.std_h),
    color=ColorSchemes.seaborn_colorblind[3],
    linewidth=2.5,
)


# Add axis for hyperfitness ECDF
ax = Axis(
    gl_ecdf[1, 2],
    xlabel="ground truth fitness |z-score|",
    ylabel="ECDF",
    aspect=AxisAspect(1)
)

# Group data by replicate
df_group = DF.groupby(df_fitness, :rep)

# Loop through replicate
for (i, data) in enumerate(df_group)
    # Extract replicate
    rep = first(data.rep)
    # Plot hyperfitness ECDF
    ecdfplot!(
        ax,
        abs.((data.mean .- data.fitness) ./ data.std),
        label="$(rep)",
        linewidth=2.5,
    )
end # for

# Add legend
axislegend(ax, position=:rb)

fig

# ---------------------------------------------------------------------------- #

println("Plot posterior predictive checks for a few barcodes...")
# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [2, 3]

# Extract barcodes
bc_ids = DF.sort(
    unique(df_counts[.!(df_counts.neutral), [:barcode, :fitness]]),
    :fitness
)
# List example barcodes to plot
bc_plot = StatsBase.sample(
    bc_ids.barcode, n_row * n_col, ordered=true, replace=false
)

# Define colors
colors = [
    get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
    get(ColorSchemes.Oranges_9, LinRange(0.5, 1, length(qs))),
]

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add GridLayout
        gl = gl_ppc[row, col] = GridLayout()
        # Add axis
        local ax = [Axis(gl[i, 1:6]) for i = 1:length(unique(df_counts.rep))]

        if (col == 1) & (row == 1)
            # Loop through replicates
            for (rep, (key, value)) in enumerate(sort(rep_vars))
                # the posterior predictive checks
                param = Dict(
                    :population_mean_fitness => :s̲ₜ,
                    :population_std_fitness => :logσ̲ₜ,
                )
                # Compute posterior predictive checks
                ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
                    df_samples[:, value], n_ppc; param=param
                )

                # Plot posterior predictive checks
                BayesFitUtils.viz.ppc_time_series!(
                    ax[rep], qs, ppc_mat;
                    colors=colors[rep], time=sort(unique(df_counts.time))[2:end]
                )

                # Plot log-frequency ratio of neutrals
                BayesFitUtils.viz.logfreq_ratio_time_series!(
                    ax[rep],
                    df_counts[(df_counts.neutral).&(df_counts.rep.==string(key)), :];
                    freq_col=:freq,
                    color=:black,
                    alpha=0.5,
                    linewidth=2
                )

                # Add title
                ax[rep].title = "$key | neutrals"
                ax[rep].titlesize = 12
            end # for
        else
            # Loop through replicates
            for (rep, (key, value)) in enumerate(sort(rep_vars))

                # Extract data
                data_bc = DF.sort(
                    df_counts[
                        (df_counts.barcode.==bc_plot[counter]).&(df_counts.rep.==string(key)),
                        :],
                    :time
                )

                # Extract variables for barcode PPC
                vars_bc = [
                    value[occursin.("̲ₜ", value)]
                    df_advi[
                        (df_advi.id.==bc_plot[counter]).&(df_advi.rep.==string(key)),
                        :varname]
                ]
                # Extract specific mutant variables variable name
                s_var = first(df_advi[
                    (df_advi.id.==bc_plot[counter]).&(df_advi.rep.==string(key)).&(df_advi.vartype.=="bc_fitness"),
                    :varname])
                logσ_var = first(df_advi[
                    (df_advi.id.==bc_plot[counter]).&(df_advi.rep.==string(key)).&(df_advi.vartype.=="bc_std"),
                    :varname])

                # Define dictionary with corresponding parameters for variables needed
                # for the posterior predictive checks
                local param = Dict(
                    :bc_mean_fitness => Symbol(s_var),
                    :bc_std_fitness => Symbol(logσ_var),
                    :population_mean_fitness => Symbol("s̲ₜ"),
                )
                # Compute posterior predictive checks
                local ppc_mat = BarBay.stats.logfreq_ratio_bc_ppc(
                    df_samples[:, Symbol.(vars_bc)],
                    n_ppc; param=param
                )

                # Plot posterior predictive checks
                BayesFitUtils.viz.ppc_time_series!(
                    ax[rep], qs, ppc_mat;
                    colors=colors[rep], time=sort(unique(df_counts.time))[2:end]
                )

                # Plot log-frequency ratio of neutrals
                BayesFitUtils.viz.logfreq_ratio_time_series!(
                    ax[rep],
                    data_bc,
                    freq_col=:freq,
                    color=:black,
                    linewidth=3,
                    markersize=8
                )

                # Compute mean and std for fitness values
                mean_s = round(
                    StatsBase.mean(df_samples[:, s_var]), sigdigits=2
                )
                std_s = round(StatsBase.std(df_samples[:, s_var]), sigdigits=2)

                # Add title
                # ax[rep].title = L"s^{(m)} = %$(mean_s){\pm%$(std_s)}"
                ax[rep].title = "$key | s⁽ᵐ⁾= $(mean_s)±$(std_s)"
                ax[rep].titlesize = 12
            end # for

        end # if
        # Hide axis decorations
        hidedecorations!.(ax, grid=false)
        # Set row and col gaps
        rowgap!(gl, 1)

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(gl_ppc[end, :, Bottom()], "time [dilution cycles]", fontsize=20)
# Add y-axis label
Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)

# ---------------------------------------------------------------------------- #

# Add subplot labels
Label(
    gl_illustrator[1, 1, TopLeft()], "(A)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_data[1, 1, TopLeft()], "(B)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_data[1, 2, TopLeft()], "(C)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_hyper_vs_fit_true[1, 1, TopLeft()], "(D)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_ppc[1, 1, TopLeft()], "(E)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_comp[1, 1, TopLeft()], "(F)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_comp[1, 2, TopLeft()], "(G)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_ecdf[1, 1, TopLeft()], "(H)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_ecdf[1, 2, TopLeft()], "(I)",
    fontsize=24,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Change spacing between subplots
rowgap!(gl_fig, 0)
colgap!(gl_fig, 20)

save("$(git_root())/doc/figs/fig04B-I.pdf", fig)

fig
