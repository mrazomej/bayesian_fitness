##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import project package
import BayesFitUtils

# Import library package
import BayesFitness

# Import basic math
import LinearAlgebra
import StatsBase
import Distributions
import Random

# Import iterator tools
import Combinatorics

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data..\n")

# Import data
df_counts = CSV.read(
    "$(git_root())/data/abreu_2023/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Process individual datasets
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Select columns that indicate the dataset
df_idx = df_counts[:, 13:end]

# Enumerate the columns (needed to run things in parallel smoothly)
n_col = size(df_idx, 2)

# Threads.@threads 
for col = 1:n_col

    println("Extracting raw barcode counts")
    # Extract data
    data = df_counts[df_idx[:, col], 1:12]

    # Process data to arrays
    data_dict = BayesFitness.utils.data_to_arrays(
        data; rep_col=:rep, env_col=:env
    )

    # Compile directory name
    out_dir = "./output/$(first(data.condition))_$(names(df_idx)[col])"

    println(out_dir)

    # Generate figure dictionary if it doesn't exist
    fig_dir = "$(out_dir)/figs"
    if !isdir(fig_dir)
        mkdir(fig_dir)
    end # if

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Define list of environments
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Extract list of environments
    envs = data_dict[:envs]
    # Extract number of environments
    n_env = data_dict[:n_env]
    # Extract number of replicates
    n_rep = data_dict[:n_rep]
    # For multi-replicate inferences replicate the list of environments when
    # necessary
    if (n_env > 1) & (n_rep > 1) & !(typeof(envs) <: Vector{<:Vector})
        envs = repeat([envs], n_rep)
    end # if

    # Define number of environments
    println("Number of environments: $(n_env)")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Read ADVI results
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Load JLD2 file
    file = Glob.glob("$(out_dir)/advi*4500*csv")

    df_advi = CSV.read(file, DF.DataFrame)

    # Split variables by replicate
    rep_vars = Dict(
        parse(Int64, rep) => df_advi[df_advi.rep.==rep, :varname]
        for rep in unique(df_advi[(df_advi.rep.≠"N/A"), :rep])
    )


    # Rename environment when there's only one
    if n_env == 1
        df_advi = df_advi[:, DF.Not(:env)]
        df_advi[!, :env] .= first(data.env)
    end

    # Define color for environments to keep consistency
    env_colors = Dict(
        String.(unique(df_advi.env)) .=>
            ColorSchemes.tableau_10[1:length(unique(df_advi.env))]
    )


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Generate samples from distribution and format into dataframe
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Define number of samples
    n_samples = 5_000

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
    # Compare mean fitness for individual replicates
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    println("Plotting fitness comparison between replicates...")
    # Define number of replicates
    n_rep = length(unique(data.rep))
    # Collect all possible pairs of plots
    rep_pairs = sort(collect(Combinatorics.combinations(unique(data.rep), 2)))

    # Initialize figure
    fig = Figure(resolution=(350 * length(rep_pairs), 350))

    # Add axis
    ax = [
        Axis(
            fig[1, i],
            title="fitness comparison",
            aspect=AxisAspect(1)
        ) for i = 1:length(rep_pairs)
    ]

    # Loop through pairs of replicates
    for (i, p) in enumerate(rep_pairs)
        # Select replicates data
        df_advi_reps = df_advi[
            ((df_advi.rep.=="$(p[1])").|(df_advi.rep.=="$(p[2])")).&(df_advi.vartype.=="bc_fitness"),
            :]

        # Plot identity line
        lines!(
            ax[i],
            repeat([[minimum(df_advi_reps.mean) * 1.05,
                    maximum(df_advi_reps.mean) * 1.05]], 2)...,
            linestyle=:dash,
            color="black"
        )

        # Group data by environment
        df_group = DF.groupby(
            df_advi_reps[df_advi_reps.vartype.=="bc_fitness", :], :env
        )

        # Loop through environments
        for (j, d) in enumerate(df_group)
            # Group data by repeat
            data_group = DF.groupby(d, :rep)

            # Plot x-axis error bars
            errorbars!(
                ax[i],
                data_group[1].mean,
                data_group[2].mean,
                data_group[1].std,
                direction=:x,
                linewidth=1.5,
                color=(:gray, 0.5)
            )
            # Plot y-axis error bars
            errorbars!(
                ax[i],
                data_group[1].mean,
                data_group[2].mean,
                data_group[2].std,
                direction=:y,
                linewidth=1.5,
                color=(:gray, 0.5)
            )
        end # for

        # Loop through environments
        for (j, d) in enumerate(df_group)
            # Group data by repeat
            data_group = DF.groupby(d, :rep)
            # Plot fitness values
            scatter!(
                ax[i],
                data_group[1].mean,
                data_group[2].mean,
                markersize=8,
                color=(env_colors[first(data_group[1].env)], 0.3)
            )
        end # for

        # Label axis
        ax[i].xlabel = "fitness replicate R$(p[1])"
        ax[i].ylabel = "fitness replicate R$(p[2])"
    end # for

    save("$(fig_dir)/advi_fitness_comparison_replicates.pdf", fig)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Compare mean fitness with hyperparameter
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    println("Plotting comparison between replicate fitness and hyper-fitness...")
    # Initialize figure
    fig = Figure(resolution=(350 * n_rep, 350))

    # Add axis
    ax = [
        Axis(
            fig[1, i],
            xlabel="hyperparameter fitness",
            ylabel="individual replicate fitness",
            aspect=AxisAspect(1),
        ) for i = 1:n_rep
    ]

    # Loop through repeats
    for (i, rep) in enumerate(sort(unique(data.rep)))
        # Select replicates data
        df_advi_reps = df_advi[
            ((df_advi.rep.=="$rep").&(df_advi.vartype.=="bc_fitness")).|(df_advi.vartype.=="bc_hyperfitness"),
            :]
        # Plot identity line
        lines!(
            ax[i],
            repeat([[minimum(df_advi_reps.mean) * 1.05,
                    maximum(df_advi_reps.mean) * 1.05]], 2)...,
            linestyle=:dash,
            color=:black,
        )

        # Add plot title
        ax[i].title = "replicate R$(rep)"
        # Loop through environments
        for (j, env) in enumerate(sort(unique(df_advi_reps.env)))
            # Extract data
            d = df_advi_reps[(df_advi_reps.env.==env), :]
            # Plot x-axis error bars
            errorbars!(
                ax[i],
                d[d.vartype.=="bc_hyperfitness", :mean],
                d[d.vartype.=="bc_fitness", :mean],
                d[d.vartype.=="bc_hyperfitness", :std],
                direction=:x,
                linewidth=1.5,
                color=(:gray, 0.5)
            )
            # Plot y-axis error bars
            errorbars!(
                ax[i],
                d[d.vartype.=="bc_hyperfitness", :mean],
                d[d.vartype.=="bc_fitness", :mean],
                d[d.vartype.=="bc_fitness", :std],
                direction=:y,
                linewidth=1.5,
                color=(:gray, 0.5)
            )
        end # for
    end # for

    for (i, rep) in enumerate(sort(unique(data.rep)))
        # Select replicates data
        df_advi_reps = df_advi[
            ((df_advi.rep.=="$rep").&(df_advi.vartype.=="bc_fitness")).|(df_advi.vartype.=="bc_hyperfitness"),
            :]
        # Loop through environments
        for (j, env) in enumerate(sort(unique(df_advi_reps.env)))
            # Extract data
            d = df_advi_reps[df_advi_reps.env.==env, :]
            # Plot fitness values
            scatter!(
                ax[i],
                d[d.vartype.=="bc_hyperfitness", :mean],
                d[d.vartype.=="bc_fitness", :mean],
                markersize=8,
                color=(env_colors[env], 0.75)
            )
        end # for
    end # for

    save("$(fig_dir)/advi_fitness_comparison_hyperparameter.pdf", fig)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot posterior predictive checks for neutral lineages in joint inference
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    println("Plotting posteiror predictive checks for neutral lineages...")
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
    ppc_color = get(ColorSchemes.Purples_9, LinRange(0.35, 1.0, length(qs)))

    # Initialize figure
    fig = Figure(resolution=(450 * n_rep, 350))

    # Loop through replicates
    for (i, (key, value)) in enumerate(rep_vars)

        # Compute posterior predictive checks
        ppc_mat = BayesFitness.stats.logfreq_ratio_popmean_ppc(
            df_samples[:, value], n_ppc; model=:normal, param=param
        )

        # Add axis
        ax = Axis(
            fig[1, i],
            xlabel="time point",
            ylabel="ln(fₜ₊₁/fₜ)",
            title="neutral lineages PPC | R$(key)",
            backgroundcolor=:white,
        )

        # Define time-environment relation
        time_env = Matrix(
            unique(
                DF.sort(
                    data[data.rep.==values(key),
                        [:time, :env]],
                    :time
                )
            )
        )

        # Define time
        t = vec(collect(axes(ppc_mat, 2)))

        # Loop through each time point
        for t = 2:size(time_env, 1)
            # Color plot background
            vspan!(
                ax,
                time_env[t, 1] - 0.5,
                time_env[t, 1] + 0.5,
                color=(
                    env_colors[String(time_env[t, 2])], 0.25
                )
            )
        end # for

        # Define time vector again for plotting PPC
        t = Int64.(time_env[:, 1])[2:end]

        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat;
            colors=ppc_color, time=t
        )

        # Plot log-frequency ratio of neutrals
        BayesFitUtils.viz.logfreq_ratio_time_series!(
            ax,
            data[(data.neutral).&(data.rep.==parse(Int64, string(key)[end])), :];
            freq_col=:freq,
            color=:black,
            alpha=0.5,
            linewidth=2,
            markersize=8
        )

        # Set axis limits
        xlims!(ax, minimum(t) - 0.25, maximum(t) + 0.25)
    end # for
    # Save figure into pdf
    save("$(fig_dir)/advi_logfreqratio_ppc_neutral.pdf", fig)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Plot posterior predictive checks for barcodes
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    println("Plotting posterior predictive checks for mutants...")
    # Define number of posterior predictive check samples
    n_ppc = 500
    # Define quantiles to compute
    qs = [0.95, 0.675, 0.05]

    # Define number of rows and columns
    n_row, n_col = [3, 3]

    # List example barcodes to plot
    bc_plot = StatsBase.sample(
        unique(data[.!(data.neutral), :barcode]), n_row * n_col
    )

    # Define colors
    colors = [
        get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
        get(ColorSchemes.Purples_9, LinRange(0.5, 1, length(qs))),
        get(ColorSchemes.Greens_9, LinRange(0.5, 1, length(qs)))
    ]

    # Initialize figure
    fig = Figure(resolution=(300 * n_col, 450 * n_row))

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
            for rep in sort(values.(keys(rep_vars)))

                # Extract data
                data_bc = DF.sort(
                    data[
                        (data.barcode.==bc_plot[counter]).&(data.rep.==rep),
                        :],
                    :time
                )

                # Extract variables for barcode PPC
                vars_bc = [
                    rep_vars[rep][occursin.("̲ₜ", rep_vars[rep])]
                    df_advi[
                        (df_advi.id.==bc_plot[counter]).&(df_advi.rep.=="$rep"),
                        :varname]
                ]

                # Define time-environment relation
                time_env = Matrix(
                    DF.sort(unique(data_bc[:, [:time, :env]]), :time)
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

                if n_env == 1
                    # Locate fitness variable
                    s_var = first(vars_bc[occursin.("s̲⁽ᵐ⁾", vars_bc)])
                    # Define dictionary with corresponding parameters for
                    # variables needed for the posterior predictive checks
                    local param = Dict(
                        :bc_mean_fitness => Symbol(s_var),
                        :bc_std_fitness => Symbol(
                            replace(s_var, "s" => "logσ")
                        ),
                        :population_mean_fitness => Symbol("s̲ₜ"),
                    )
                    # Compute posterior predictive checks
                    local ppc_mat = BayesFitness.stats.logfreq_ratio_bc_ppc(
                        df_samples[:, Symbol.(vars_bc)],
                        n_ppc;
                        model=:normal,
                        param=param
                    )
                else
                    # Define dictionary with corresponding parameters for
                    # variables needed for the posterior predictive checks
                    local param = Dict(
                        :bc_mean_fitness => Symbol("s̲⁽ᵐ⁾"),
                        :bc_std_fitness => Symbol("logσ̲⁽ᵐ⁾"),
                        :population_mean_fitness => Symbol("s̲ₜ"),
                    )
                    # Compute posterior predictive checks
                    local ppc_mat = BayesFitness.stats.logfreq_ratio_multienv_ppc(
                        df_samples[:, Symbol.(vars_bc)],
                        n_ppc,
                        envs[rep];
                        model=:normal,
                        param=param
                    )
                end

                # Define time
                t = vec(collect(axes(ppc_mat, 2)))

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
                    linewidth=3,
                    markersize=10
                )

                # Add title
                ax[rep].title = "replicate R$rep"
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
            counter += 1
        end  # for
    end # for

    # Add x-axis label
    Label(fig[end, :, Bottom()], "time points", fontsize=20)
    # Add y-axis label
    Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)

    save("$(fig_dir)/advi_logfreqratio_ppc_mutant.pdf", fig)

end # for