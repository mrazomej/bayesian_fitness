##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import project package
import BayesFitUtils

# Import library package
import BarBay

# Import basic math
import LinearAlgebra
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
df_counts = CSV.read(
    "$(git_root())/data/logistic_growth/data_008/tidy_data.csv", DF.DataFrame
)

# Define number of experimental repeats
n_rep = length(unique(df_counts.rep))

# Define number of unique environments
n_env = length(unique(df_counts.env))

# Define list of environments
envs = [1, 1, 2, 3, 1, 2, 3]
# Define unique environments
env_unique = unique(envs)

# Define color for environments to keep consistency
env_colors = Dict(env_unique .=> ColorSchemes.tableau_10[1:3])

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = first(Glob.glob("./output/advi_meanfield*"))

# Load distribution
advi_results = JLD2.load(file)
bc_ids = advi_results["ids"]
dist = advi_results["dist"]
vars = advi_results["var"]

# Generate tidy dataframe with distribution information
df_advi = BarBay.utils.advi_to_df(
    dist, vars, bc_ids; n_rep=2, envs=[1, 1, 2, 3, 1, 2, 3]
)

# Split variables by replicate
rep_vars = Dict(
    Symbol(rep) => df_advi[df_advi.rep.==rep, :varname]
    for rep in unique(df_advi.rep)
)
# Remove "N/A" from dictionary
delete!(rep_vars, Symbol("N/A"))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples
n_samples = 10_000

# Sample from ADVI joint distribution and convert to dataframe
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
# Compare median fitness for individual replicates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="replicate 1 fitness",
    ylabel="replicate 2 fitness",
    title="fitness comparison",
    aspect=AxisAspect(1)
)

# Plot identity line
lines!(ax, repeat([[-0.75, 2.5]], 2)..., linestyle=:dash, color="black")

# Define colors
colors = ColorSchemes.seaborn_colorblind

# Group data by environment
df_group = DF.groupby(df_advi[df_advi.var_type.=="bc_fitness", :], :env)

# Loop through environments
for (i, data) in enumerate(df_group)
    # Group data by repeat
    data_group = DF.groupby(data, :rep)

    # Plot x-axis error bars
    errorbars!(
        ax,
        data_group[1].mean,
        data_group[2].mean,
        data_group[1].std,
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.25)
    )
    # Plot y-axis error bars
    errorbars!(
        ax,
        data_group[1].mean,
        data_group[2].mean,
        data_group[2].std,
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.1)
    )
end # for

# Loop through environments
for (i, data) in enumerate(df_group)
    # Group data by repeat
    data_group = DF.groupby(data, :rep)
    # Plot fitness values
    scatter!(
        ax,
        data_group[1].mean,
        data_group[2].mean,
        markersize=5,
        color=(colors[i], 0.75)
    )
end # for

save("./output/figs/advi_fitness_comparison_replicates.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness with hyperparameter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350 * 2, 350))

# Add axis
ax = [
    Axis(
        fig[1, i],
        xlabel="hyper parameter fitness",
        ylabel="individual replicate fitness",
        aspect=AxisAspect(1),
    ) for i = 1:2
]

# Define plot colors
colors = ColorSchemes.seaborn_colorblind

# Loop through repeats
for rep = 1:n_rep
    # Plot identity line
    lines!(
        ax[rep], repeat([[-0.75, 2.5]], 2)..., linestyle=:dash, color="black"
    )

    # Add plot title
    ax[rep].title = "replicate R$(rep)"

    # Loop through environments
    for env = 1:n_env
        # Extract data
        data = df_advi[df_advi.env.==env, :]
        # Plot x-axis error bars
        errorbars!(
            ax[rep],
            data[data.var_type.=="bc_hyperfitness", :mean],
            data[
                (data.var_type.=="bc_fitness").&(data.rep.=="R$rep"),
                :mean],
            data[data.var_type.=="bc_hyperfitness", :std],
            direction=:x,
            linewidth=1.5,
            color=(:gray, 0.25)
        )
        # Plot y-axis error bars
        errorbars!(
            ax[rep],
            data[data.var_type.=="bc_hyperfitness", :mean],
            data[
                (data.var_type.=="bc_fitness").&(data.rep.=="R$rep"),
                :mean],
            data[
                (data.var_type.=="bc_fitness").&(data.rep.=="R$rep"),
                :std],
            direction=:y,
            linewidth=1.5,
            color=(:gray, 0.1)
        )
    end # for
end # for

for rep = 1:n_rep
    # Loop through environments
    for env = 1:n_env
        # Extract data
        data = df_advi[df_advi.env.==env, :]
        # Plot fitness values
        scatter!(
            ax[rep],
            data[data.var_type.=="bc_hyperfitness", :mean],
            data[
                (data.var_type.=="bc_fitness").&(data.rep.=="R$rep"),
                :mean],
            markersize=5,
            color=(colors[env], 0.75)
        )
    end # for
end # for

save("./output/figs/advi_fitness_comparison_hyperparameter.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Extract information
data_advi = df_advi[
    df_advi.var_type.=="bc_hyperfitness", [:id, :mean, :std, :env]
]
DF.rename!(data_advi, :id => :barcode)

data_counts = df_counts[
    (.!df_counts.neutral).&(df_counts.rep.=="R1"), [:barcode, :fitness, :env]
]

# Combine information
data = DF.leftjoin(data_advi, data_counts; on=[:barcode, :env])

# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="true hyper-fitness value",
    ylabel="ADVI inferred hyper-fitness",
)

# Plot identity line
lines!(
    ax, repeat([[-0.1, 2.5]], 2)..., linestyle=:dash, color="black"
)

# Plot errorbars
errorbars!(
    ax,
    data.fitness,
    data.mean,
    data.std,
    direction=:y,
    color=(:gray, 0.5)
)

# Plot mean comparision
scatter!(ax, data.fitness, data.mean, markersize=8)


save("./output/figs/advi_fitness_true_hyperparameter.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|mean - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(data.mean .- data.fitness))

save("./output/figs/advi_median_true_ecdf.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF for true fitness z-score 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute Z-score of true fitness values
fitness_zscore = (data.fitness .- data.mean) ./ data.std
# Initialize figure
fig = Figure(resolution=(400, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|z-score|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(fitness_zscore))

save("./output/figs/advi_zscore_ecdf.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for neutral lineages in joint inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define dictionary with corresponding parameters for variables needed for
# the posterior predictive checks
param = Dict(
    :population_mean_fitness => :s̲ₜ,
    :population_std_fitness => :logσ̲ₜ,
)

# Define number of posterior predictive check samples
n_ppc = 500

# Define quantiles to compute
qs = [0.05, 0.68, 0.95]

# Define colors
ppc_color = get(ColorSchemes.Purples_9, LinRange(0.25, 1.0, length(qs)))

# Initialize figure
fig = Figure(resolution=(450 * 2, 350))

# Loop through replicates
for (i, (key, value)) in enumerate(rep_vars)

    # Compute posterior predictive checks
    ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
        df_samples[:, value], n_ppc; model=:normal, param=param
    )

    # Add axis
    ax = Axis(
        fig[1, i],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="neutral lineages PPC | $(key)",
        backgroundcolor=:white,
    )

    # Define time-environment relation
    time_env = Matrix(
        unique(df_counts[df_counts.rep.==string(key), [:time, :env]])
    )

    # Define time
    t = vec(collect(axes(ppc_mat, 2)) .+ 1)

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
        ax, qs, ppc_mat; colors=ppc_color, time=t
    )

    # Plot log-frequency ratio of neutrals
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax,
        df_counts[(df_counts.neutral).&(df_counts.rep.==string(key)), :];
        freq_col=:freq,
        color=:black,
        alpha=0.5,
        linewidth=2,
        markersize=8
    )

    # Set axis limits
    xlims!(ax, 1.75, 7.25)

end # for
# Save figure into pdf
save("./output/figs/advi_logfreqratio_ppc_neutral.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for barcodes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [3, 3]

# List example barcodes to plot
bc_plot = StatsBase.sample(bc_ids, n_row * n_col)

# Define colors
colors = [
    get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
    get(ColorSchemes.Purples_9, LinRange(0.5, 1, length(qs)))
]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add GridLayout
        gl = fig[row, col] = GridLayout()
        # Add axis
        ax = [Axis(gl[i, 1:6], backgroundcolor=:white) for i = 1:n_rep]

        # Loop through replicates
        for (rep, (key, value)) in enumerate(rep_vars)

            # Extract data
            data_bc = DF.sort(
                df_counts[
                    (df_counts.barcode.==bc_plot[counter]).&(df_counts.rep.==string(key)),
                    :],
                :time
            )

            # Extract variables for barcode PPC
            global vars_bc = [
                value[occursin.("̲ₜ", value)]
                df_advi[
                    (df_advi.id.==bc_plot[counter]).&(df_advi.rep.==string(key)),
                    :varname]
            ]

            # Define time-environment relation
            time_env = Matrix(
                unique(data_bc[data_bc.rep.==string(key), [:time, :env]])
            )

            # Loop through each time point
            for t = 2:size(time_env, 1)
                # Color plot background
                vspan!(
                    ax[rep],
                    time_env[t, 1] - 0.5,
                    time_env[t, 1] + 0.5,
                    color=(env_colors[time_env[t, 2]], 0.25)
                )
            end # for

            # Define dictionary with corresponding parameters for variables needed
            # for the posterior predictive checks
            local param = Dict(
                :mutant_mean_fitness => Symbol("s̲⁽ᵐ⁾"),
                :mutant_std_fitness => Symbol("logσ̲⁽ᵐ⁾"),
                :population_mean_fitness => Symbol("s̲ₜ"),
            )
            # Compute posterior predictive checks
            local ppc_mat = BarBay.stats.logfreq_ratio_multienv_ppc(
                df_samples[:, Symbol.(vars_bc)],
                n_ppc,
                envs;
                model=:normal,
                param=param
            )

            # Define time
            t = vec(collect(axes(ppc_mat, 2)) .+ 1)

            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax[rep], qs, ppc_mat; colors=colors[rep], time=t
            )

            # Plot log-frequency ratio of neutrals
            BayesFitUtils.viz.logfreq_ratio_time_series!(
                ax[rep],
                data_bc,
                freq_col=:freq,
                color=:black,
                linewidth=2,
                markersize=8
            )

            # Add title
            ax[rep].title = "replicate $key"
            ax[rep].titlesize = 12
        end # for

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)
        # Set row and col gaps
        rowgap!(gl, 1)

        # Add barcode as title
        Label(
            gl[0, 3:4],
            text="barcode $(bc_plot[counter])",
            fontsize=12,
            justification=:center,
            lineheight=0.9
        )

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=20)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)

save("./output/figs/advi_logfreqratio_ppc_mutant.pdf", fig)

fig