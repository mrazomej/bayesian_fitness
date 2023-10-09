##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

import Revise
# Import project package
import BayesFitUtils
# Import package for Bayesian inference
import BayesFitness

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import Glob

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

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

# Generate figure dictionary
if !isdir("./output/figs/")
    mkdir("./output/figs/")
end # if

# Define directory for this specific analysis
fig_dir = "./output/figs/advi_hierarchicalreplicate_inference/"

# Generate directory for single dataset inference
if !isdir(fig_dir)
    mkdir(fig_dir)
end # if

println("Loading data...")

# Load raw reads
df_counts = CSV.read(
    "$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame
)

# Read datasets visual evaluation info
df_include = CSV.read(
    "$(git_root())/data/kinsler_2020/exp_include.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Listing ADVI files and metadata
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define directory
out_dir = "$(git_root())/code/processing/kinsler_2020/output/" *
          "advi_meanfield_hierarchicalreplicate_inference"

# List files
files = Glob.glob("$(out_dir)/*csv"[2:end], "/")

# Initialize dataframe to save ADVI results
df_advi = DF.DataFrame()
# Initialize dataframe to save bc fitness values
df_fitness = DF.DataFrame()

# Loop through files
for f in files
    # Split file name
    f_split = split(split(f, "/")[end], "_")

    # Extract file metadata
    env = replace(f_split[4], "env" => "")
    n_samples = parse(Int64, replace(f_split[5], "samples" => ""))
    n_steps = parse(Int64, replace(f_split[6], "steps.csv" => ""))

    # Load ADVI results into memory removing redundant columns
    df_tmp = CSV.read(f, DF.DataFrame)[:, DF.Not([:env])]

    # Add metadata information
    DF.insertcols!(
        df_tmp,
        :env .=> env,
        :n_samples .=> n_samples,
        :n_steps .=> n_steps
    )

    # Add to df_advi
    DF.append!(df_advi, df_tmp)

    # Extract bc fitness values
    df_fit = df_tmp[(df_tmp.vartype.=="bc_fitness"), :]

    # Extract and append hyperfitness values for each fitness value
    DF.leftjoin!(
        df_fit,
        DF.rename(
            df_tmp[(df_tmp.vartype.=="bc_hyperfitness"),
                [:mean, :std, :varname, :id]],
            :mean => :mean_h,
            :std => :std_h,
            :varname => :varname_h
        );
        on=:id
    )

    # Add to df_advi
    DF.append!(df_fitness, df_fit)
end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plotting PPC 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Group data by environment and replicate
df_group = DF.groupby(df_advi, [:env])

println("Plotting first environment results")

# Extract data 
df_advi_env = df_group[1]

# Extract metadata
env = first(df_advi_env.env)

# Split variables by replicate
rep_vars = Dict(
    Symbol(rep) => df_advi_env[df_advi_env.rep.==rep, :varname]
    for rep in unique(df_advi_env.rep)
)
# Remove "N/A" from dictionary
delete!(rep_vars, Symbol("N/A"))

# Define number of samples
n_samples = 2_500

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

# Extract include information
df_in = df_include[df_include.env.==env, :]

# Initialize dataframe to save data
data = DF.DataFrame()

# Loop through datasets
for row in eachrow(df_in)
    # Extract dataset
    d = df_counts[(df_counts.env.==row.env).&(df_counts.rep.==row.rep), :]
    # Check if first time point should be removed
    if row.rm_T0
        # Remove first time point if required in df_include
        d = d[d.time.>minimum(d.time), :]
    end # if
    # Append dataframes
    DF.append!(data, d)
end # for

# Extract fitness inference
df_fit = df_fitness[(df_fitness.env.==env), :]

# Initialize figure
fig = Figure(resolution=(600, 900))

# Add grid layout for posterior predictive checks
gl_ppc = fig[1, 1] = GridLayout()

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [4, 4]

# List example barcodes to plot
bc_plot = StatsBase.sample(
    eachrow(DF.sort(df_fit, :mean)),
    n_row * n_col,
    replace=false,
    ordered=true
)

# Extract IDs
bc_ids = [bc.id for bc in bc_plot]

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
        local ax = [Axis(gl[i, 1:6]) for i = 1:length(unique(data.rep))]

        if (col == 1) & (row == 1)
            # Loop through replicates
            for (rep, (key, value)) in enumerate(sort(rep_vars))
                # the posterior predictive checks
                param = Dict(
                    :population_mean_fitness => :s̲ₜ,
                    :population_std_fitness => :logσ̲ₜ,
                )
                # Compute posterior predictive checks
                ppc_mat = BayesFitness.stats.logfreq_ratio_popmean_ppc(
                    df_samples[:, value], n_ppc; model=:normal, param=param
                )

                # Plot posterior predictive checks
                BayesFitUtils.viz.ppc_time_series!(
                    ax[rep], qs, ppc_mat;
                    colors=colors[rep], time=sort(unique(data.time))[2:end]
                )

                # Plot log-frequency ratio of neutrals
                BayesFitUtils.viz.logfreq_ratio_time_series!(
                    ax[rep],
                    data[(data.neutral).&(data.rep.==string(key)), :];
                    freq_col=:freq,
                    color=:black,
                    alpha=0.5,
                    linewidth=2
                )

                # Add title
                ax[rep].title = "$key"
                ax[rep].titlesize = 10

                # Add title
                Label(gl[0, :], "neutral barcodes", fontsize=10)
            end # for
        else
            # Loop through replicates
            for (rep, (key, value)) in enumerate(sort(rep_vars))

                # Extract data
                data_bc = DF.sort(
                    data[
                        (string.(data.barcode).==bc_ids[counter]).&(data.rep.==string(key)),
                        :],
                    :time
                )

                # Extract variables for barcode PPC
                vars_bc = [
                    value[occursin.("̲ₜ", value)]
                    df_advi_env[
                        (df_advi_env.id.==bc_ids[counter]).&(df_advi_env.rep.==string(key)),
                        :varname]
                ]
                # Extract specific mutant variables variable name
                s_var = first(df_advi_env[
                    (df_advi_env.id.==bc_ids[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_fitness"),
                    :varname])
                logσ_var = first(df_advi_env[
                    (df_advi_env.id.==bc_ids[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_std"),
                    :varname])

                # Define dictionary with corresponding parameters for variables needed
                # for the posterior predictive checks
                local param = Dict(
                    :bc_mean_fitness => Symbol(s_var),
                    :bc_std_fitness => Symbol(logσ_var),
                    :population_mean_fitness => Symbol("s̲ₜ"),
                )
                # Compute posterior predictive checks
                local ppc_mat = BayesFitness.stats.logfreq_ratio_bc_ppc(
                    df_samples[:, Symbol.(vars_bc)],
                    n_ppc;
                    model=:normal,
                    param=param
                )

                # Plot posterior predictive checks
                BayesFitUtils.viz.ppc_time_series!(
                    ax[rep], qs, ppc_mat;
                    colors=colors[rep], time=sort(unique(data.time))[2:end]
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
                Label(gl[0, :], "bc $(bc_ids[counter])", fontsize=10)
                # Add title
                ax[rep].title = "$key | s⁽ᵐ⁾= $(mean_s)±$(std_s)"
                ax[rep].titlesize = 10
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
Label(gl_ppc[end, :, Bottom()], "time points", fontsize=15)
# Add y-axis label
Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=15)
# Add title
Label(gl_ppc[0, :], "$(env)", fontsize=15)

save("$(fig_dir)/ppc_$(env)env.pdf", fig)

println("Plot the remainder in parallel")

# Threads.@threads 
for d = 2:length(df_group)
    # Define quantiles to compute
    qs = [0.95, 0.675, 0.05]

    # Define colors
    colors = [
        get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
        get(ColorSchemes.Oranges_9, LinRange(0.5, 1, length(qs))),
    ]

    # Extract data 
    df_advi_env = df_group[d]

    # Extract metadata
    env = first(df_advi_env.env)

    println("$(env)")

    # Split variables by replicate
    rep_vars = Dict(
        Symbol(rep) => df_advi_env[df_advi_env.rep.==rep, :varname]
        for rep in unique(df_advi_env.rep)
    )
    # Remove "N/A" from dictionary
    delete!(rep_vars, Symbol("N/A"))

    # Define number of samples
    n_samples = 2_500

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

    # Extract include information
    df_in = df_include[df_include.env.==env, :]

    # Initialize dataframe to save data
    data = DF.DataFrame()

    # Loop through datasets
    for row in eachrow(df_in)
        # Extract dataset
        d = df_counts[(df_counts.env.==row.env).&(df_counts.rep.==row.rep), :]
        # Check if first time point should be removed
        if row.rm_T0
            # Remove first time point if required in df_include
            d = d[d.time.>minimum(d.time), :]
        end # if
        # Append dataframes
        DF.append!(data, d)
    end # for

    # Extract fitness inference
    df_fit = df_fitness[(df_fitness.env.==env), :]

    # Initialize figure
    fig = Figure(resolution=(600, 900))

    # Add grid layout for posterior predictive checks
    gl_ppc = fig[1, 1] = GridLayout()

    # Define number of posterior predictive check samples
    n_ppc = 500
    # Define quantiles to compute
    qs = [0.95, 0.675, 0.05]

    # Define number of rows and columns
    n_row, n_col = [4, 4]

    # List example barcodes to plot
    bc_plot = StatsBase.sample(
        eachrow(DF.sort(df_fit, :mean)),
        n_row * n_col,
        replace=false,
        ordered=true
    )

    # Extract IDs
    bc_ids = [bc.id for bc in bc_plot]

    # Initialize plot counter
    counter = 1
    # Loop through rows
    for row in 1:n_row
        # Loop through columns
        for col in 1:n_col
            # Add GridLayout
            gl = gl_ppc[row, col] = GridLayout()
            # Add axis
            local ax = [Axis(gl[i, 1:6]) for i = 1:length(colors)]

            if (col == 1) & (row == 1)
                # Loop through replicates
                for (rep, (key, value)) in enumerate(sort(rep_vars))
                    # Check that only two replicates are plot
                    if rep > length(colors)
                        continue
                    end # if
                    # the posterior predictive checks
                    param = Dict(
                        :population_mean_fitness => :s̲ₜ,
                        :population_std_fitness => :logσ̲ₜ,
                    )
                    # Compute posterior predictive checks
                    ppc_mat = BayesFitness.stats.logfreq_ratio_popmean_ppc(
                        df_samples[:, value], n_ppc; model=:normal, param=param
                    )

                    # Plot posterior predictive checks
                    BayesFitUtils.viz.ppc_time_series!(
                        ax[rep], qs, ppc_mat;
                        colors=colors[rep],
                        time=sort(
                            unique(data[(data.rep.==string(key)), :time])
                        )[2:size(ppc_mat, 2)+1]
                    )

                    # Plot log-frequency ratio of neutrals
                    BayesFitUtils.viz.logfreq_ratio_time_series!(
                        ax[rep],
                        data[(data.neutral).&(data.rep.==string(key)), :];
                        freq_col=:freq,
                        color=:black,
                        alpha=0.5,
                        linewidth=2
                    )

                    # Add title
                    ax[rep].title = "$key"
                    ax[rep].titlesize = 10

                    # Add title
                    Label(gl[0, :], "neutral barcodes", fontsize=10)
                end # for
            else
                # Loop through replicates
                for (rep, (key, value)) in enumerate(sort(rep_vars))
                    # Check that only two replicates are plot
                    if rep > length(colors)
                        continue
                    end # if

                    # Extract data
                    data_bc = DF.sort(
                        data[
                            (string.(data.barcode).==bc_ids[counter]).&(data.rep.==string(key)),
                            :],
                        :time
                    )

                    # Extract variables for barcode PPC
                    vars_bc = [
                        value[occursin.("̲ₜ", value)]
                        df_advi_env[
                            (df_advi_env.id.==bc_ids[counter]).&(df_advi_env.rep.==string(key)),
                            :varname]
                    ]
                    # Extract specific mutant variables variable name
                    s_var = first(df_advi_env[
                        (df_advi_env.id.==bc_ids[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_fitness"),
                        :varname])
                    logσ_var = first(df_advi_env[
                        (df_advi_env.id.==bc_ids[counter]).&(df_advi_env.rep.==string(key)).&(df_advi_env.vartype.=="bc_std"),
                        :varname])

                    # Define dictionary with corresponding parameters for variables needed
                    # for the posterior predictive checks
                    local param = Dict(
                        :bc_mean_fitness => Symbol(s_var),
                        :bc_std_fitness => Symbol(logσ_var),
                        :population_mean_fitness => Symbol("s̲ₜ"),
                    )
                    # Compute posterior predictive checks
                    local ppc_mat = BayesFitness.stats.logfreq_ratio_bc_ppc(
                        df_samples[:, Symbol.(vars_bc)],
                        n_ppc;
                        model=:normal,
                        param=param
                    )

                    # Plot posterior predictive checks
                    BayesFitUtils.viz.ppc_time_series!(
                        ax[rep], qs, ppc_mat;
                        colors=colors[rep],
                        time=sort(unique(data_bc.time))[2:size(ppc_mat, 2)+1]
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
                    Label(gl[0, :], "bc $(bc_ids[counter])", fontsize=10)
                    # Add title
                    ax[rep].title = "$key | s⁽ᵐ⁾= $(mean_s)±$(std_s)"
                    ax[rep].titlesize = 10
                end # for

            end # if
            # Hide axis decorations
            hidedecorations!.(ax, grid=false)
            # Set row and col gaps
            rowgap!(gl, 1)

            # Update counter
            counter += 1
        end  # for
    end # for

    # Add x-axis label
    Label(gl_ppc[end, :, Bottom()], "time points", fontsize=15)
    # Add y-axis label
    Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=15)
    # Add title
    Label(gl_ppc[0, :], "$(env)", fontsize=15)

    save("$(fig_dir)/ppc_$(env)env.pdf", fig)
end # for