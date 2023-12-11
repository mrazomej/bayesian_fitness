##
println("Loading packages...")



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

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

Random.seed!(42)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load the data
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

println("Loading data...")

# Import data
df = CSV.read(
    "$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame
)

# Define datasets to process
envs = ["1.7%", "1.8%", "2ugFlu", "8.5uMGdA"]

# Define titles
env_titles = ["1.7% glucose", "1.8% glucose", "2 µg Flu", "8.5 µM Gda"]

# Keep data from environments to process
df = df[[x ∈ envs for x in df.env], :]

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot environment trajectories
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Group data by environment
df_group = DF.groupby(df, :env)

# Initialize figure
fig = Figure(resolution=(300 * 4, 350 * 2))

# An array to store the GridLayout objects
gls = []

# Generating four GridLayout objects
for i in 1:2, j in 1:2
    # Initialize grid layout
    gl = GridLayout()
    # Storing each GridLayout in the array
    push!(gls, gl)
    # Set GridLayout position
    fig.layout[i, j] = gl
end

# Loop through groups
for (i, data) in enumerate(df_group)
    # Group data by replicates
    data_group = DF.groupby(data, :rep)
    # Loop through each replicate
    for (j, d) in enumerate(data_group)
        # Set axis for plot
        ax = Axis(
            gls[i][1, j],
            xlabel="time [dilution cycles]",
            ylabel="barcode frequency",
            aspect=AxisAspect(1),
            yscale=log10,
            title=first(d.rep)
        )

        # Plot mutant barcode trajectories
        BayesFitUtils.viz.bc_time_series!(
            ax,
            d[.!(d.neutral), :],
            quant_col=:freq,
            zero_lim=10^-7.5,
            alpha=0.3
        )

        # Plot neutral barcode trajectories
        BayesFitUtils.viz.bc_time_series!(
            ax,
            d[d.neutral, :],
            quant_col=:freq,
            zero_lim=10^-7.5,
            color=ColorSchemes.Blues_9[end],
        )
    end # for

    # Set plot title based on environment
    Label(
        gls[i][:, :, Top()],
        env_titles[i],
        fontsize=22,
        padding=(0, 0, 30, 0)
    )

    # Add plot label
    Label(
        gls[i][1, 1, TopLeft()], "($(collect('A':'D')[i]))",
        fontsize=26,
        padding=(0, 5, 5, 0),
        halign=:right
    )
end # for

save("$(git_root())/doc/figs/figSIX_kinsler_data.pdf", fig)
save("$(git_root())/doc/figs/figSIX_kinsler_data.png", fig)

fig

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load single dataset ADVI results
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

println("Loading single dataset ADVI results")

# Define output directory
advidir = "$(git_root())/code/processing/data014_kinsler2020/output/" *
          "advi_meanfield_joint_inference"

# List files
files = Glob.glob("$(advidir)/*csv"[2:end], "/")

# Initialize dataframe
df_advi = DF.DataFrame()

# Loop through files
for f in files
    # Split file name
    fname = split(split(f, "/")[end], "_")
    # Extract environment
    env = replace(fname[2], "env" => "")
    # Extract replicate
    rep = replace(fname[3], "rep" => "")

    # Read dataframe
    df_tmp = CSV.read(f, DF.DataFrame)

    # Add env and rep column
    df_tmp[!, :env] .= env
    df_tmp[!, :rep] .= rep

    append!(df_advi, df_tmp)
end # for

# Keep only fitness values
df_fitness_single = df_advi[df_advi.vartype.=="bc_fitness", :]

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot replicates comparisons
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Group data by environment
df_group = DF.groupby(df_fitness_single, :env)

# Initialize figure
fig = Figure(resolution=(300 * 2, 300 * 2))

# An array to store the GridLayout objects
gls = []
# An array to store axis
ax = []

# Generating four GridLayout objects
for i in 1:2, j in 1:2
    # Initialize grid layout
    gl = GridLayout()
    # Storing each GridLayout in the array
    push!(gls, gl)

    # Set GridLayout position
    fig.layout[i, j] = gl

    # Add axis
    axs = Axis(
        gl[1, 1],
        xlabel="fitness replicate R1",
        ylabel="fitness replicate R2",
        aspect=AxisAspect(1)
    )
    # Storing each Axis in the array
    push!(ax, axs)
end

# Loop through groups
for (i, data) in enumerate(df_group)
    # Group data by replicate
    data_group = DF.groupby(data, :rep)

    # Plot identity line
    lines!(
        ax[i],
        repeat([[minimum(data.mean), maximum(data.mean)]], 2)...;
        color=:black,
        linestyle=:dash
    )

    # Add x-axis error bars
    errorbars!(
        ax[i],
        DF.sort!(data_group[1], :id).mean,
        DF.sort!(data_group[2], :id).mean,
        DF.sort!(data_group[1], :id).std,
        color=(:gray, 0.5),
        direction=:x
    )

    # Add y-axis error bars
    errorbars!(
        ax[i],
        DF.sort!(data_group[1], :id).mean,
        DF.sort!(data_group[2], :id).mean,
        DF.sort!(data_group[2], :id).std,
        color=(:gray, 0.5),
        direction=:y
    )

    # Plot comparison
    scatter!(
        ax[i],
        DF.sort!(data_group[1], :id).mean,
        DF.sort!(data_group[2], :id).mean,
    )

    # Add title
    ax[i].title = env_titles[i]

    # Add plot label
    Label(
        gls[i][1, 1, TopLeft()], "($(collect('A':'D')[i]))",
        fontsize=22,
        padding=(0, 5, 5, 0),
        halign=:right
    )
end # for

save("$(git_root())/doc/figs/figSIX_kinsler_rep_single.pdf", fig)
save("$(git_root())/doc/figs/figSIX_kinsler_rep_signle.png", fig)

fig

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot PPC for single datset inferences
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Define number of samples
n_samples = 4_000

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [2, 2]

# Extract barcodes
bc_ids = sort(unique(df[.!(df.neutral), :barcode]))

# List example barcodes to plot
bc_plot = StatsBase.sample(
    bc_ids, n_row * n_col, ordered=true, replace=false
)

# Define colors
colors = [
    get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
    get(ColorSchemes.Oranges_9, LinRange(0.5, 1, length(qs))),
]

# Initialize figure
fig = Figure(resolution=(300 * 4, 350 * 3))

# An array to store the GridLayout objects
gls = []

# Generating four GridLayout objects
for i in 1:2, j in 1:2
    # Initialize grid layout
    gl = GridLayout()
    # Storing each GridLayout in the array
    push!(gls, gl)
    # Set GridLayout position
    fig.layout[i, j] = gl
end

# Loop through environments
for (i, env) in enumerate(envs)
    # Extract data
    df_counts = df[df.env.==env, :]

    # Extract ADVI inference
    df_advi_env = df_advi[df_advi.env.==env, :]

    # Sample from posterior MvNormal
    df_samples = DF.DataFrame(
        Random.rand(
            Distributions.MvNormal(
                df_advi_env.mean, LinearAlgebra.Diagonal(df_advi_env.std .^ 2)
            ),
            n_samples
        )',
        df_advi_env.varname .* df_advi_env.rep
    )

    # Initialize plot counter
    counter = 1

    # Split variables by replicate
    rep_vars = Dict(
        Symbol(rep) => df_advi_env[df_advi_env.rep.==rep, :varname] .* rep
        for rep in unique(df_advi_env.rep)
    )

    # Loop through rows and columns
    for row in 1:n_row, col in 1:n_col
        # Add GridLayout
        gl = gls[i][row, col] = GridLayout()
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
                    df_samples[:, value],
                    n_ppc;
                    param=param
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
                    df_advi_env[
                        (df_advi_env.id.==string.(bc_plot)[counter]).&(df_advi_env.rep.==string(key)),
                        :varname] .* string(key)
                ]
                # Extract specific mutant variables variable name
                s_var = first(df_advi_env[
                    (df_advi_env.id.==string.(bc_plot)[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_fitness"),
                    :varname]) .* string(key)
                logσ_var = first(df_advi_env[
                    (df_advi_env.id.==string.(bc_plot)[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_std"),
                    :varname]) .* string(key)

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
                    n_ppc;
                    param=param
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
        counter += 1
    end  # for

    # Set plot title based on environment
    Label(
        gls[i][:, :, Top()],
        env_titles[i],
        fontsize=22,
        padding=(0, 0, 30, 0)
    )

    # Add plot label
    Label(
        gls[i][1, 1, TopLeft()], "($(collect('A':'D')[i]))",
        fontsize=26,
        padding=(0, 5, 5, 0),
        halign=:right
    )
end # for

# Add x-axis label
Label(gls[3][end, :, Bottom()], "time [dilution cycles]", fontsize=20)
Label(gls[4][end, :, Bottom()], "time [dilution cycles]", fontsize=20)

# Add y-axis label
Label(
    gls[1][:, 1, Left()],
    "ln(fₜ₊₁/fₜ)",
    rotation=π / 2,
    fontsize=20,
)
Label(
    gls[3][:, 1, Left()],
    "ln(fₜ₊₁/fₜ)",
    rotation=π / 2,
    fontsize=20,
)

save("$(git_root())/doc/figs/figSIX_kinsler_ppc_single.pdf", fig)
save("$(git_root())/doc/figs/figSIX_kinsler_ppc_single.png", fig)

fig


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load hierarchical ADVI results
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

println("Loading hierarchical ADVI results")

# Define output directory
advidir = "$(git_root())/code/processing/data014_kinsler2020/output/" *
          "advi_meanfield_hierarchicalreplicate_inference"

# List files
files = Glob.glob("$(advidir)/*csv"[2:end], "/")

# Initialize dataframe
df_advi = DF.DataFrame()

# Loop through files
for f in files
    # Split file name
    fname = split(split(f, "/")[end], "_")
    # Extract environment
    env = replace(fname[4], "01samples" => "")

    # Read dataframe
    df_tmp = CSV.read(f, DF.DataFrame)

    # Add env and rep column
    df_tmp[!, :env] .= env

    append!(df_advi, df_tmp)
end # for

# Keep only fitness values
df_fitness_hier = df_advi[df_advi.vartype.=="bc_fitness", :]

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot replicates comparisons
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Group data by environment
df_group = DF.groupby(df_fitness_hier, :env)

# Initialize figure
fig = Figure(resolution=(300 * 2, 300 * 2))

# An array to store the GridLayout objects
gls = []
# An array to store axis
ax = []

# Generating four GridLayout objects
for i in 1:2, j in 1:2
    # Initialize grid layout
    gl = GridLayout()
    # Storing each GridLayout in the array
    push!(gls, gl)

    # Set GridLayout position
    fig.layout[i, j] = gl

    # Add axis
    axs = Axis(
        gl[1, 1],
        xlabel="fitness replicate R1",
        ylabel="fitness replicate R2",
        aspect=AxisAspect(1)
    )
    # Storing each Axis in the array
    push!(ax, axs)
end

# Loop through groups
for (i, data) in enumerate(df_group)
    # Group data by replicate
    data_group = DF.groupby(data, :rep)

    # Plot identity line
    lines!(
        ax[i],
        repeat([[minimum(data.mean), maximum(data.mean)] .* 1.25], 2)...;
        color=:black,
        linestyle=:dash
    )

    # Add x-axis error bars
    errorbars!(
        ax[i],
        DF.sort!(data_group[1], :id).mean,
        DF.sort!(data_group[2], :id).mean,
        DF.sort!(data_group[1], :id).std,
        color=(:gray, 0.5),
        direction=:x
    )

    # Add y-axis error bars
    errorbars!(
        ax[i],
        DF.sort!(data_group[1], :id).mean,
        DF.sort!(data_group[2], :id).mean,
        DF.sort!(data_group[2], :id).std,
        color=(:gray, 0.5),
        direction=:y
    )

    # Plot comparison
    scatter!(
        ax[i],
        DF.sort!(data_group[1], :id).mean,
        DF.sort!(data_group[2], :id).mean,
    )

    # Add title
    ax[i].title = env_titles[i]

    # Add plot label
    Label(
        gls[i][1, 1, TopLeft()], "($(collect('A':'D')[i]))",
        fontsize=22,
        padding=(0, 5, 5, 0),
        halign=:right
    )
end # for

save("$(git_root())/doc/figs/figSIX_kinsler_rep_hier.pdf", fig)
save("$(git_root())/doc/figs/figSIX_kinsler_rep_hier.png", fig)

fig

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot PPC for hierarchical inferences
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Initialize figure
fig = Figure(resolution=(300 * 4, 350 * 3))

# An array to store the GridLayout objects
gls = []

# Generating four GridLayout objects
for i in 1:2, j in 1:2
    # Initialize grid layout
    gl = GridLayout()
    # Storing each GridLayout in the array
    push!(gls, gl)
    # Set GridLayout position
    fig.layout[i, j] = gl
end

# Loop through environments
for (i, env) in enumerate(envs)
    # Extract data
    df_counts = df[df.env.==env, :]

    # Extract ADVI inference
    df_advi_env = df_advi[df_advi.env.==env, :]

    # Sample from posterior MvNormal
    df_samples = DF.DataFrame(
        Random.rand(
            Distributions.MvNormal(
                df_advi_env.mean, LinearAlgebra.Diagonal(df_advi_env.std .^ 2)
            ),
            n_samples
        )',
        df_advi_env.varname
    )

    # Initialize plot counter
    counter = 1

    # Split variables by replicate
    rep_vars = Dict(
        Symbol(rep) => df_advi_env[df_advi_env.rep.==rep, :varname]
        for rep in unique(df_advi_env.rep)
    )
    # Remove "N/A" from dictionary
    delete!(rep_vars, Symbol("N/A"))

    # Loop through rows and columns
    for row in 1:n_row, col in 1:n_col
        # Add GridLayout
        gl = gls[i][row, col] = GridLayout()
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
                    df_advi_env[
                        (df_advi_env.id.==string.(bc_plot)[counter]).&(df_advi_env.rep.==string(key)),
                        :varname]
                ]
                # Extract specific mutant variables variable name
                s_var = first(df_advi_env[
                    (df_advi_env.id.==string.(bc_plot)[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_fitness"),
                    :varname])
                logσ_var = first(df_advi_env[
                    (df_advi_env.id.==string.(bc_plot)[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_std"),
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
                    n_ppc;
                    param=param
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
        counter += 1
    end  # for

    # Set plot title based on environment
    Label(
        gls[i][:, :, Top()],
        env_titles[i],
        fontsize=22,
        padding=(0, 0, 30, 0)
    )

    # Add plot label
    Label(
        gls[i][1, 1, TopLeft()], "($(collect('A':'D')[i]))",
        fontsize=26,
        padding=(0, 5, 5, 0),
        halign=:right
    )
end # for

# Add x-axis label
Label(gls[3][end, :, Bottom()], "time [dilution cycles]", fontsize=20)
Label(gls[4][end, :, Bottom()], "time [dilution cycles]", fontsize=20)

# Add y-axis label
Label(
    gls[1][:, 1, Left()],
    "ln(fₜ₊₁/fₜ)",
    rotation=π / 2,
    fontsize=20,
)
Label(
    gls[3][:, 1, Left()],
    "ln(fₜ₊₁/fₜ)",
    rotation=π / 2,
    fontsize=20,
)

save("$(git_root())/doc/figs/figSIX_kinsler_ppc_hier.pdf", fig)
save("$(git_root())/doc/figs/figSIX_kinsler_ppc_hier.png", fig)

fig