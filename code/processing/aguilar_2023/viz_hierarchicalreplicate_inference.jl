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

# Import iterator tools
import Combinatorics

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
BayesFitUtils.viz.theme_makie!()

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

println("Loading data...")

# Import data
df = CSV.read(
    "$(git_root())/data/aguilar_2023/tidy_counts_oligo.csv", DF.DataFrame
)
# Remove repeat "N/A"
df = df[df.rep.!="N/A", :]

# Add oligos by edit
df_counts = DF.combine(
    DF.groupby(df, [:edit, :rep, :time, :neutral]), :count => sum
)

# Rename column
DF.rename!(df_counts, :count_sum => :count)

# Compute sum of each edit
df_sum = DF.combine(DF.groupby(df_counts, [:rep, :time]), :count => sum)

# Add sum to dataframe
DF.leftjoin!(df_counts, df_sum; on=[:rep, :time])

# Compute frequency
df_counts[!, :freq] = df_counts.count ./ df_counts.count_sum

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results...")
# Define file
file = first(Glob.glob("./output/advi_meanfield*hierarchicalreplicate*"))

# Convert results to tidy dataframe
df_advi = CSV.read(file, DF.DataFrame)

# Split variables by replicate
rep_vars = Dict(
    Symbol(rep) => df_advi[df_advi.rep.==rep, :varname]
    for rep in unique(df_advi.rep)
)
# Remove "N/A" from dictionary
delete!(rep_vars, Symbol("N/A"))

# Define number of replicates
n_rep = length(rep_vars)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples
n_samples = 3_000

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
# Compare mean fitness for individual replicates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting fitness comparison between replicates...")
# Collect all possible pairs of plots
rep_pairs = collect(Combinatorics.combinations(unique(df_counts.rep), 2))

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

# Define colors
colors = ColorSchemes.seaborn_colorblind

# Loop through pairs of replicates
for (i, p) in enumerate(rep_pairs)

    # Plot identity line
    lines!(ax[i], repeat([[-1, 0.75]], 2)..., linestyle=:dash, color="black")

    # Group data by repeat
    data_group = DF.groupby(
        df_advi[
            ((df_advi.rep.==p[1]).|(df_advi.rep.==p[2])).&(df_advi.vartype.=="bc_fitness"),
            :],
        :rep
    )

    # Plot x-axis error bars
    errorbars!(
        ax[i],
        data_group[1].mean,
        data_group[2].mean,
        data_group[1].std,
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.25)
    )
    # Plot y-axis error bars
    errorbars!(
        ax[i],
        data_group[1].mean,
        data_group[2].mean,
        data_group[2].std,
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.25)
    )

    # Plot fitness values
    scatter!(
        ax[i],
        data_group[1].mean,
        data_group[2].mean,
        markersize=7,
    )

    # Label axis
    ax[i].xlabel = "fitness replicate $(p[1])"
    ax[i].ylabel = "fitness replicate $(p[2])"
end # for

save("./output/figs/advi_fitness_comparison_replicates.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare mean fitness with hyperparameter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting comparison between replicate fitness and fitness hyperparameter")

# Initialize figure
fig = Figure(resolution=(350 * n_rep, 350))

# Add axis
ax = [
    Axis(
        fig[1, i],
        xlabel="hyper parameter fitness",
        ylabel="individual replicate fitness",
        aspect=AxisAspect(1),
    ) for i = 1:n_rep
]

# Define plot colors
colors = ColorSchemes.seaborn_colorblind

# Loop through repeats
for rep in 1:n_rep
    # Plot identity line
    lines!(
        ax[rep], repeat([[-1, 0.75]], 2)..., linestyle=:dash, color="black"
    )

    # Add plot title
    ax[rep].title = "replicate R$(rep)"

    # Plot x-axis error bars
    errorbars!(
        ax[rep],
        df_advi[df_advi.vartype.=="bc_hyperfitness", :mean],
        df_advi[
            (df_advi.vartype.=="bc_fitness").&(df_advi.rep.=="R$rep"),
            :mean],
        df_advi[df_advi.vartype.=="bc_hyperfitness", :std],
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.25)
    )
    # Plot y-axis error bars
    errorbars!(
        ax[rep],
        df_advi[df_advi.vartype.=="bc_hyperfitness", :mean],
        df_advi[
            (df_advi.vartype.=="bc_fitness").&(df_advi.rep.=="R$rep"),
            :mean],
        df_advi[
            (df_advi.vartype.=="bc_fitness").&(df_advi.rep.=="R$rep"),
            :std],
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.25)
    )

    # Plot fitness values
    scatter!(
        ax[rep],
        df_advi[df_advi.vartype.=="bc_hyperfitness", :mean],
        df_advi[
            (df_advi.vartype.=="bc_fitness").&(df_advi.rep.=="R$rep"),
            :mean],
        markersize=5,
    )
end # for

save("./output/figs/advi_fitness_comparison_hyperparameter.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for neutral lineages in joint inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting posterior predictive checks for neutral lineages...")
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
ppc_color = get(ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs)))

# Initialize figure
fig = Figure(resolution=(400 * n_rep, 350))

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
        title="neutral lineages PPC | $(key)",
    )

    # Plot posterior predictive checks
    BayesFitUtils.viz.ppc_time_series!(
        ax, qs, ppc_mat;
        colors=ppc_color,
        time=sort(
            unique(df_counts[(df_counts.rep.==string(key)), :time])
        )[2:end]
    )

    # Plot log-frequency ratio of neutrals
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax,
        df_counts[(df_counts.neutral).&(df_counts.rep.==string(key)), :];
        freq_col=:freq,
        color=:black,
        alpha=0.5,
        linewidth=2,
        id_col=:edit
    )

end # for
# Save figure into pdf
save("./output/figs/advi_logfreqratio_ppc_neutral.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for barcodes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plot posterior predictive checks for a few barcodes...")
# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [3, 3]

# Extract barcodes
bc_ids = unique(df_counts[.!(df_counts.neutral), :edit])
# List example barcodes to plot
bc_plot = StatsBase.sample(bc_ids, n_row * n_col)

# Define colors
colors = [
    get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
    get(ColorSchemes.Purples_9, LinRange(0.5, 1, length(qs))),
    get(ColorSchemes.Greens_9, LinRange(0.5, 1, length(qs)))
]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 400 * n_row))

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add GridLayout
        gl = fig[row, col] = GridLayout()
        # Add axis
        ax = [Axis(gl[i, 1:6]) for i = 1:n_rep]

        # Loop through replicates
        for (rep, (key, value)) in enumerate(rep_vars)

            # Extract data
            global data_bc = DF.sort(
                df_counts[
                    (df_counts.edit.==bc_plot[counter]).&(df_counts.rep.==string(key)),
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
                time=sort(
                    unique(df_counts[df_counts.rep.==string(key), :time])
                )[2:end]
            )

            # Plot log-frequency ratio of neutrals
            BayesFitUtils.viz.logfreq_ratio_time_series!(
                ax[rep],
                data_bc,
                freq_col=:freq,
                color=:black,
                linewidth=3,
                markersize=8,
                id_col=:edit,
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
            text="edit $(bc_plot[counter])",
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