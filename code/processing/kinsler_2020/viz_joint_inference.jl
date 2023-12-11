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
fig_dir = "./output/figs/advi_meanfield_joint_inference/"

# Generate directory for single dataset inference
if !isdir(fig_dir)
    mkdir(fig_dir)
end # if

println("Loading data...")

# 
df_counts = CSV.read(
    "$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Listing ADVI files and metadata
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results")

# Define directory
out_dir = "./output/advi_meanfield_joint_inference"

# List files
files = Glob.glob("$(out_dir)/*csv")

# Initialize dataframe to save ADVI results
df_advi = DF.DataFrame()

# Loop through files
for f in files
    # Split file name
    f_split = split(split(f, "/")[end], "_")

    # Extract file metadata
    env = replace(f_split[2], "env" => "")
    rep = replace(f_split[3], "rep" => "")
    rmT0 = parse(Bool, replace(f_split[4], "rmT0" => ""))
    n_samples = parse(Int64, replace(f_split[5], "samples" => ""))
    n_steps = parse(Int64, replace(f_split[6], "steps.csv" => ""))

    # Load ADVI results into memory removing redundant columns
    df_tmp = CSV.read(f, DF.DataFrame)[:, DF.Not([:rep, :env])]

    # Add metadata information
    DF.insertcols!(
        df_tmp,
        :env .=> env,
        :rep .=> rep,
        :rm_T0 .=> rmT0,
        :n_samples .=> n_samples,
        :n_steps .=> n_steps
    )

    # Add to df_advi
    DF.append!(df_advi, df_tmp)
end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plotting PPC 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting posterior predictive checks")

# Group data by environment and replicate
df_group = DF.groupby(df_advi, [:env, :rep])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate single plot so that we can run the rest in parallel
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Generating first plot...")
# Extract data 
df_env = df_group[1]

# Extract metadata
env = first(df_env.env)
rep = first(df_env.rep)
rm_T0 = first(df_env.rm_T0)

# Extract mutant fitness
df_fitness = df_env[(df_env.vartype.=="bc_fitness"), :]

println("$(env) | $(rep)")

# Define number of samples
n_samples = 2_500

# Sample from posterior MvNormal
df_samples = DF.DataFrame(
    Random.rand(
        Distributions.MvNormal(
            df_env.mean, LinearAlgebra.Diagonal(df_env.std .^ 2)
        ),
        n_samples
    )',
    df_env.varname
)

# Extract data
data = df_counts[
    (df_counts.env.==first(df_env.env)).&(df_counts.rep.==first(df_env.rep)),
    :]

# Remove T0 if necessary
if rm_T0
    data = data[(data.time.>first(sort(unique(data.time)))), :]
end # if


# Initialize figure
fig = Figure(resolution=(600, 600))

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
                df_samples, n_ppc; model=:normal, param=param
            )

            if rm_T0
                # Define time
                t = vec(collect(axes(ppc_mat, 2)) .+ 1)
            else
                t = vec(collect(axes(ppc_mat, 2)) .+ 0)
            end # if

            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax, qs, ppc_mat; colors=colors, time=t
            )

            # Plot log-frequency ratio of neutrals
            BayesFitUtils.viz.logfreq_ratio_time_series!(
                ax,
                data[data.neutral, :];
                time_col=:time,
                freq_col=:freq,
                color=:black,
                alpha=1.0,
                linewidth=1.5
            )

            # Hide axis decorations
            hidedecorations!.(ax, grid=false)

            ax.title = "neutral lineages"
            ax.titlesize = 10

            global counter += 1

            continue
        end # if

        # Extract data
        data_bc = DF.sort(
            data[string.(data.barcode).==bc_plot[counter].id, :], :time
        )

        # Define colors
        local colors = get(
            ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs))
        )

        # Define dictionary with corresponding parameters for variables
        # needed for the posterior predictive checks
        local param = Dict(
            :bc_mean_fitness => Symbol(bc_plot[counter].varname),
            :bc_std_fitness => Symbol(
                replace(bc_plot[counter].varname, "s" => "logσ")
            ),
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        local ppc_mat = BarBay.stats.logfreq_ratio_bc_ppc(
            df_samples, n_ppc; model=:normal, param=param
        )
        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors
        )

        # Add scatter of data
        scatterlines!(
            ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.0
        )

        # Define fitness ranges to display in title
        vals = [
            round(bc_plot[counter].mean; sigdigits=2),
            round(bc_plot[counter].std; sigdigits=2),
        ]

        # Add title
        ax.title = "bc $(bc_plot[counter].id) | s⁽ᵐ⁾= $(vals[1])±$(vals[2])"
        ax.titlesize = 10

        ## == Plot format == ##

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(gl_ppc[end, :, Bottom()], "time points", fontsize=15)
# Add y-axis label
Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=15)
# Add title
Label(gl_ppc[0, :], "$(env) | $(rep)", fontsize=15)
# Set spacing
rowgap!(gl_ppc, 0)
colgap!(gl_ppc, 4)

save("$(fig_dir)/ppc_$(env)env_$(rep)rep_$(rm_T0)rmT0.pdf", fig)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot in parallel
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting rest in parallel...")
# Loop through datasets
Threads.@threads for d = 2:length(df_group)
    # Extract data 
    df_env = df_group[d]

    # Extract metadata
    env = first(df_env.env)
    rep = first(df_env.rep)
    rm_T0 = first(df_env.rm_T0)

    # Extract mutant fitness
    df_fitness = df_env[(df_env.vartype.=="bc_fitness"), :]

    println("$(env) | $(rep)")

    # Define number of samples
    n_samples = 2_500

    # Sample from posterior MvNormal
    df_samples = DF.DataFrame(
        Random.rand(
            Distributions.MvNormal(
                df_env.mean, LinearAlgebra.Diagonal(df_env.std .^ 2)
            ),
            n_samples
        )',
        df_env.varname
    )

    # Extract data
    data = df_counts[
        (df_counts.env.==first(df_env.env)).&(df_counts.rep.==first(df_env.rep)),
        :]

    # Remove T0 if necessary
    if rm_T0
        data = data[(data.time.>first(sort(unique(data.time)))), :]
    end # if


    # Initialize figure
    fig = Figure(resolution=(600, 600))

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
                    df_samples, n_ppc; model=:normal, param=param
                )

                if rm_T0
                    # Define time
                    t = vec(collect(axes(ppc_mat, 2)) .+ 1)
                else
                    t = vec(collect(axes(ppc_mat, 2)) .+ 0)
                end # if

                # Plot posterior predictive checks
                BayesFitUtils.viz.ppc_time_series!(
                    ax, qs, ppc_mat; colors=colors, time=t
                )

                # Plot log-frequency ratio of neutrals
                BayesFitUtils.viz.logfreq_ratio_time_series!(
                    ax,
                    data[data.neutral, :];
                    time_col=:time,
                    freq_col=:freq,
                    color=:black,
                    alpha=1.0,
                    linewidth=1.5
                )

                # Hide axis decorations
                hidedecorations!.(ax, grid=false)

                ax.title = "neutral lineages"
                ax.titlesize = 10

                counter += 1

                continue
            end # if

            # Extract data
            data_bc = DF.sort(
                data[string.(data.barcode).==bc_plot[counter].id, :], :time
            )

            # Define colors
            local colors = get(
                ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs))
            )

            # Define dictionary with corresponding parameters for variables
            # needed for the posterior predictive checks
            local param = Dict(
                :bc_mean_fitness => Symbol(bc_plot[counter].varname),
                :bc_std_fitness => Symbol(
                    replace(bc_plot[counter].varname, "s" => "logσ")
                ),
                :population_mean_fitness => :s̲ₜ,
            )
            # Compute posterior predictive checks
            local ppc_mat = BarBay.stats.logfreq_ratio_bc_ppc(
                df_samples, n_ppc; model=:normal, param=param
            )
            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax, qs, ppc_mat; colors=colors
            )

            # Add scatter of data
            scatterlines!(
                ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.0
            )

            # Define fitness ranges to display in title
            vals = [
                round(bc_plot[counter].mean; sigdigits=2),
                round(bc_plot[counter].std; sigdigits=2),
            ]

            # Add title
            ax.title = "bc $(bc_plot[counter].id) | s⁽ᵐ⁾= $(vals[1])±$(vals[2])"
            ax.titlesize = 10

            ## == Plot format == ##

            # Hide axis decorations
            hidedecorations!.(ax, grid=false)

            # Update counter
            counter += 1
        end  # for
    end # for

    # Add x-axis label
    Label(gl_ppc[end, :, Bottom()], "time points", fontsize=15)
    # Add y-axis label
    Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=15)
    # Add title
    Label(gl_ppc[0, :], "$(env) | $(rep)", fontsize=15)
    # Set spacing
    rowgap!(gl_ppc, 0)
    colgap!(gl_ppc, 4)

    save("$(fig_dir)/ppc_$(env)env_$(rep)rep_$(rm_T0)rmT0.pdf", fig)

end # for