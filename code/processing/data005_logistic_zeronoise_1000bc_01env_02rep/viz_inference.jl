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
    "$(git_root())/data/logistic_growth/data_005/tidy_data.csv", DF.DataFrame
)

# Define number of experimental repeats
n_rep = length(unique(df_counts.rep))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = first(Glob.glob("./output/advi_meanfield*3000*"))

# Load distribution
advi_results = JLD2.load(file)
ids_advi = advi_results["ids"]
dist_advi = advi_results["dist"]
var_advi = advi_results["var"]

# Convert results to tidy dataframe
df_advi = BayesFitness.utils.advi2df(dist_advi, var_advi, ids_advi; n_rep=2)

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
    lines!(ax[i], repeat([[-0.25, 1.75]], 2)..., linestyle=:dash, color="black")

    # Group data by repeat
    data_group = DF.groupby(
        df_advi[
            ((df_advi.rep.==p[1]).|(df_advi.rep.==p[2])).&(df_advi.vartype.=="mut_fitness"),
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
        color=(:gray, 0.1)
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
        ax[rep], repeat([[-0.25, 1.75]], 2)..., linestyle=:dash, color="black"
    )

    # Add plot title
    ax[rep].title = "replicate R$(rep)"

    # Plot x-axis error bars
    errorbars!(
        ax[rep],
        df_advi[df_advi.vartype.=="mut_hyperfitness", :mean],
        df_advi[
            (df_advi.vartype.=="mut_fitness").&(df_advi.rep.=="R$rep"),
            :mean],
        df_advi[df_advi.vartype.=="mut_hyperfitness", :std],
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.25)
    )
    # Plot y-axis error bars
    errorbars!(
        ax[rep],
        df_advi[df_advi.vartype.=="mut_hyperfitness", :mean],
        df_advi[
            (df_advi.vartype.=="mut_fitness").&(df_advi.rep.=="R$rep"),
            :mean],
        df_advi[
            (df_advi.vartype.=="mut_fitness").&(df_advi.rep.=="R$rep"),
            :std],
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.1)
    )

    # Plot fitness values
    scatter!(
        ax[rep],
        df_advi[df_advi.vartype.=="mut_hyperfitness", :mean],
        df_advi[
            (df_advi.vartype.=="mut_fitness").&(df_advi.rep.=="R$rep"),
            :mean],
        markersize=5,
    )
end # for

save("./output/figs/advi_fitness_comparison_hyperparameter.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Extract information
data_advi = df_advi[
    df_advi.vartype.=="mut_hyperfitness", [:id, :mean, :std]
]
DF.rename!(data_advi, :id => :barcode)

data_counts = df_counts[
    (.!df_counts.neutral).&(df_counts.rep.=="R1"), [:barcode, :fitness]
]

# Combine information
data = DF.leftjoin(data_advi, data_counts; on=:barcode)

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
    ax, repeat([[-0.1, 1.75]], 2)..., linestyle=:dash, color="black"
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
scatter!(
    ax,
    data.fitness,
    data.mean,
    markersize=8,
)

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
ecdfplot!(ax, abs.(data.mean .- data.fitness), label="all", color=:black)

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
ecdfplot!(ax, abs.(fitness_zscore), color=:black, label="all")

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
        colors=ppc_color, time=sort(unique(df_counts.time))[2:end]
    )

    # Plot log-frequency ratio of neutrals
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax,
        df_counts[(df_counts.neutral).&(df_counts.rep.==string(key)), :];
        freq_col=:freq,
        color=:black,
        alpha=0.5,
        linewidth=2
    )

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
bc_plot = StatsBase.sample(ids_advi, n_row * n_col)

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
            # Extract specific mutant variables variable name
            s_var = first(df_advi[
                (df_advi.id.==bc_plot[counter]).&(df_advi.rep.==string(key)).&(df_advi.vartype.=="mut_fitness"),
                :varname])
            logσ_var = first(df_advi[
                (df_advi.id.==bc_plot[counter]).&(df_advi.rep.==string(key)).&(df_advi.vartype.=="mut_error"),
                :varname])

            # Define dictionary with corresponding parameters for variables needed
            # for the posterior predictive checks
            local param = Dict(
                :mutant_mean_fitness => Symbol(s_var),
                :mutant_std_fitness => Symbol(logσ_var),
                :population_mean_fitness => Symbol("s̲ₜ"),
            )
            # Compute posterior predictive checks
            local ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
                df_samples[:, Symbol.(vars_bc)],
                n_ppc;
                model=:normal,
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