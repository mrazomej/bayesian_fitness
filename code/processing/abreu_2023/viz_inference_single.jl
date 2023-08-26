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
import LinearAlgebra

# Import libraries to manipulate data
import DataFrames as DF
import CSV

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

println("Loading data..\n")

# Import data
df_counts = CSV.read(
    "$(git_root())/data/abreu_2023/GHN_data.csv", DF.DataFrame
)

# Compute total reads per replicate and time point
df_sum = DF.combine(DF.groupby(df_counts, [:replicate, :time]), :count => sum)

# Add total sum column to dataframe
DF.leftjoin!(df_counts, df_sum; on=[:replicate, :time])

# Compute frequencies
df_counts[!, :freq] = df_counts[:, :count] ./ df_counts[:, :count_sum]

# Define number of unique environments
n_env = length(unique(df_counts.environment))

# Define environment cycles
envs = collect(unique(df_counts[:, [:time, :environment]])[:, :environment])

# Define unique environments
env_unique = unique(envs)

# Define color for environments to keep consistency
colors = Dict(env_unique .=> ColorSchemes.tableau_10[1:3])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
files = Glob.glob("./output/advi_meanfield_R*csv")


# Looop through files
for file in files
    # Extract replicate number
    rep = parse(Int64, replace(split(file, "_")[3], "R" => ""))

    # Extract data
    data = df_counts[df_counts.replicate.==rep, :]

    # Read ADVI results
    df_advi = CSV.read(file, DF.DataFrame)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Generate samples from distribution and format into dataframe
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Define number of samples
    n_samples = 10_000

    # Sample from ADVI joint distribution and convert to dataframe
    df_samples = DF.DataFrame(
        Random.rand(
            Distributions.MvNormal(
                df_advi.mean,
                LinearAlgebra.Diagonal(df_advi.std .^ 2)
            ),
            n_samples
        )',
        df_advi.varname
    )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot posterior predictive checks for neutral lineages in joint inference
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_popmean_ppc(
        df_samples, n_ppc; model=:normal, param=param
    )

    # Initialize figure
    fig = Figure(resolution=(450, 350))

    # Add axis
    ax = Axis(
        fig[1, 1],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="neutral lineages PPC",
        backgroundcolor=:white,
    )

    # Define time-environment relation
    time_env = Matrix(unique(data[:, [:time, :environment]]))

    # Define time
    t = vec(collect(axes(ppc_mat, 2)))

    # Loop through each time point
    for t = 2:size(time_env, 1)
        # Color plot background
        vspan!(
            ax,
            time_env[t, 1] - 0.5,
            time_env[t, 1] + 0.5,
            color=(colors[time_env[t, 2]], 0.25)
        )
    end # for

    # Plot posterior predictive checks
    BayesFitUtils.viz.ppc_time_series!(
        ax, qs, ppc_mat; colors=ppc_color, time=t
    )

    # Plot log-frequency ratio of neutrals
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax,
        data[data.neutral, :];
        freq_col=:freq,
        color=:black,
        alpha=0.5,
        linewidth=2,
        markersize=8
    )

    # Set axis limits
    xlims!(ax, 0.75, 5.25)

    # Save figure into pdf
    save("./output/figs/advi_logfreqratio_ppc_neutral_$rep.pdf", fig)

    fig


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot example posterior predictive checks
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Define number of posterior predictive check samples
    n_ppc = 500
    # Define quantiles to compute
    qs = [0.95, 0.675, 0.05]

    # Define number of rows and columns
    n_row, n_col = [4, 4]

    # Initialize figure
    fig = Figure(resolution=(300 * n_col, 300 * n_row))

    # List example barcodes to plot
    bc_plot = StatsBase.sample(
        unique(df_advi[(df_advi.vartype.=="mut_fitness"), :id]), n_row * n_col
    )

    # Initialize plot counter
    counter = 1
    # Loop through rows
    for row in 1:n_row
        # Loop through columns
        for col in 1:n_col
            # Add axis
            local ax = Axis(fig[row, col], backgroundcolor=:white)

            # Extract data
            data_bc = DF.sort(data[data.barcode.==bc_plot[counter], :], :time)
            # Extract fitness variable names
            s_var = df_advi[
                (df_advi.id.==bc_plot[counter]).&(df_advi.vartype.=="mut_fitness"),
                :varname]
            # Extract logσ variable names
            σ_var = replace.(s_var, Ref("s" => "logσ"))
            # Extract mean fitness variables
            sₜ_var = df_advi[(df_advi.vartype.=="pop_mean"), :varname]

            # Extract samples
            df_bc = df_samples[:, [sₜ_var; s_var; σ_var]]

            # Define colors
            local ppc_color = get(
                ColorSchemes.Purples_9, LinRange(0.5, 1.0, length(qs))
            )

            # Define dictionary with corresponding parameters for variables needed
            # for the posterior predictive checks
            local param = Dict(
                :mutant_mean_fitness => :s̲⁽ᵐ⁾,
                :mutant_std_fitness => :logσ̲⁽ᵐ⁾,
                :population_mean_fitness => :s̲ₜ,
            )
            # Compute posterior predictive checks
            local ppc_mat = BayesFitness.stats.logfreq_ratio_multienv_ppc(
                df_bc, n_ppc, envs; model=:normal, param=param
            )

            # Define time-environment relation
            time_env = Matrix(unique(data_bc[:, [:time, :environment]]))

            # Define time
            t = vec(collect(axes(ppc_mat, 2)))

            # Loop through each time point
            for t = 2:size(time_env, 1)
                # Color plot background
                vspan!(
                    ax,
                    time_env[t, 1] - 0.5,
                    time_env[t, 1] + 0.5,
                    color=(colors[time_env[t, 2]], 0.25)
                )
            end # for

            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax, qs, ppc_mat; colors=ppc_color, time=t
            )

            # Add scatter of data
            scatterlines!(
                ax, t, diff(log.(data_bc.freq)), color=:black, linewidth=2.5
            )

            # Add title
            ax.title = "barcode #$(data_bc[1, :BCID_x])"
            ax.titlesize = 18

            ## == Plot format == ##

            # Hide axis decorations
            hidedecorations!.(ax, grid=false)

            # Update counter
            counter += 1
        end  # for
    end # for

    # Add x-axis label
    Label(fig[end, :, Bottom()], "time points", fontsize=22)
    # Add y-axis label
    Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=22)

    save("./output/figs/advi_logfreqratio_ppc_mutant_$rep.pdf", fig)

    fig
end # for