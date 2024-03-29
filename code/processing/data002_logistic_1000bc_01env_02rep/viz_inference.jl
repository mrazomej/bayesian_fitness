##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import project package
import BayesFitUtils

# Import library package
import BarBay

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
df_counts = CSV.read(
    "$(git_root())/data/logistic_growth/data_002/tidy_data.csv", DF.DataFrame
)

# Define number of experimental repeats
n_rep = length(unique(df_counts.rep))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results...")
# Define file
file = first(Glob.glob("./output/advi_meanfield*5000*"))

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

# Loop through pairs of replicates
for (i, p) in enumerate(rep_pairs)

    # Plot identity line
    lines!(
        ax[i],
        repeat(
            [[
                minimum(
                    df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
                ) * 1.05,
                maximum(
                    df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
                ) * 1.05,]],
            2
        )...,
        linestyle=:dash,
        color="black"
    )

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

# Loop through repeats
for rep in 1:n_rep
    # Plot identity line
    lines!(
        ax[rep],
        repeat(
            [[
                minimum(
                    df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
                ) * 1.05,
                maximum(
                    df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
                ) * 1.05,]],
            2
        )...,
        linestyle=:dash,
        color="black"
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
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting comparison between inferred fitness and ground truth...")
# Extract information
data_advi = df_advi[
    df_advi.vartype.=="bc_hyperfitness", [:id, :mean, :std]
]
DF.rename!(data_advi, :id => :barcode)

data_counts = df_counts[
    (.!df_counts.neutral).&(df_counts.rep.=="R1"),
    [:barcode, :hyperfitness]
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
    ax,
    repeat(
        [[
            minimum(
                df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
            ) * 1.05,
            maximum(
                df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
            ) * 1.05,]],
        2
    )...,
    linestyle=:dash,
    color="black"
)

# Plot errorbars
errorbars!(
    ax,
    data.hyperfitness,
    data.mean,
    data.std,
    direction=:y,
    color=(:gray, 0.5)
)

# Plot mean comparision
scatter!(
    ax,
    data.hyperfitness,
    data.mean,
    markersize=8,
)

save("./output/figs/advi_fitness_true_hyperparameter.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting replicate inferred fitness with true fitness...")

# Extract information
data_advi = df_advi[
    df_advi.vartype.=="bc_fitness", [:id, :mean, :std, :rep]
]
DF.rename!(data_advi, :id => :barcode)

data_counts = df_counts[(.!df_counts.neutral), [:barcode, :fitness, :rep]]

# Combine information
data = DF.leftjoin(data_advi, data_counts; on=[:barcode, :rep])


# Initialize figure
fig = Figure(resolution=(350 * n_rep, 350))

# Add axis
ax = [
    Axis(
        fig[1, i],
        title="fitness comparison",
        aspect=AxisAspect(1)
    ) for i = 1:n_rep
]

# Loop through pairs of replicates
for (i, p) in enumerate(unique(data.rep))

    # Plot identity line
    lines!(
        ax[i],
        repeat(
            [[
                minimum(
                    df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
                ) * 1.05,
                maximum(
                    df_advi[df_advi.vartype.=="bc_hyperfitness", :mean]
                ) * 1.05,]],
            2
        )...,
        linestyle=:dash,
        color="black"
    )

    # Plot y-axis error bars
    errorbars!(
        ax[i],
        data[data.rep.==p, :fitness],
        data[data.rep.==p, :mean],
        data[data.rep.==p, :std],
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.25)
    )

    # Plot fitness values
    scatter!(
        ax[i],
        data[data.rep.==p, :fitness],
        data[data.rep.==p, :mean],
        markersize=7,
    )

    # Label axis
    ax[i].xlabel = "true replicate fitness"
    ax[i].ylabel = "inferred replicate fitness"
    ax[i].title = "replicate $p"
end # for

save("./output/figs/advi_fitness_comparison_replicates_truth.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting diagionstic ECDFs...")
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
colors = [
    get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
    get(ColorSchemes.Purples_9, LinRange(0.5, 1, length(qs))),
]

# Initialize figure
fig = Figure(resolution=(400 * n_rep, 350))

# Loop through replicates
for (i, (key, value)) in enumerate(sort(rep_vars))

    # Compute posterior predictive checks
    ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
        df_samples[:, value], n_ppc; model=:normal, param=param
    )

    # Add axis
    local ax = Axis(
        fig[1, i],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
        title="neutral lineages PPC | $(key)",
    )

    # Plot posterior predictive checks
    BayesFitUtils.viz.ppc_time_series!(
        ax, qs, ppc_mat;
        colors=colors[i], time=sort(unique(df_counts.time))[2:end]
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

println("Plot posterior predictive checks for a few barcodes...")
# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [3, 3]

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
    get(ColorSchemes.Purples_9, LinRange(0.5, 1, length(qs))),
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
        local ax = [Axis(gl[i, 1:6]) for i = 1:n_rep]

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

            # Compute mean and std for fitness values
            mean_s = round(StatsBase.mean(df_samples[:, s_var]), sigdigits=2)
            std_s = round(StatsBase.std(df_samples[:, s_var]), sigdigits=2)

            # Add title
            ax[rep].title = "replicate $key | s⁽ᵐ⁾= $(mean_s)±$(std_s)"
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results for individual replicates...")

# Define file
files = sort(Glob.glob("./output/advi_meanfield*rep*5000*"))

# Initialize empty dataframe
df_advi_single = DF.DataFrame()

# Loop through files
for file in files
    # Extract replicate information
    rep = replace(split(file, "_")[3], "rep" => "")

    # read dataframe
    df_tmp = CSV.read(file, DF.DataFrame)
    # Replace replicate column
    df_tmp[:, :rep] .= rep

    # Append to DataFrame
    DF.append!(df_advi_single, df_tmp)
end # for

# Extract and append hyperfitness values for each fitness value
DF.leftjoin!(
    df_fitness,
    DF.rename(
        df_advi_single[(df_advi_single.vartype.=="bc_fitness"),
            [:mean, :std, :varname, :id, :rep]],
        :mean => :mean_single,
        :std => :std_single,
        :varname => :varname_single
    );
    on=[:id, :rep]
)

# Add column with mean between replicates for singled dataset model
DF.leftjoin!(
    df_fitness,
    DF.combine(
        DF.groupby(df_fitness, :id), :mean_single => StatsBase.mean
    );
    on=:id
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between single-replicate fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350 * 2, 350 * 2))

# Add axis
ax = [
    Axis(
        fig[i, j],
        aspect=AxisAspect(1)
    ) for i = 1:2 for j = 1:2
]

# -----------------------------------------------------------------------------

# Add identity line
lines!(
    ax[1],
    repeat([[1.75, -0.5,]], 2)...,
    linestyle=:dash,
    color="black"
)

# Plot erorr bars for fitness comparison between replicates in hierarchical
# model
errorbars!(
    ax[1],
    df_fitness[(df_fitness.rep.=="R1"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :mean],
    df_fitness[(df_fitness.rep.=="R1"), :std],
    direction=:x,
    color=(:gray, 0.5)
)
errorbars!(
    ax[1],
    df_fitness[(df_fitness.rep.=="R1"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :std],
    direction=:y,
    color=(:gray, 0.25)
)

# Plot mean for fitness comparison between replicates in hierarchical model
scatter!(
    ax[1],
    df_fitness[(df_fitness.rep.=="R1"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :mean],
    markersize=5
)

ax[1].title = "hierarchical model"
ax[1].xlabel = "replicate R1 fitness"
ax[1].ylabel = "replicate R2 fitness"

# -----------------------------------------------------------------------------

# Add identity line
lines!(
    ax[2],
    repeat([[1.75, -0.5,]], 2)...,
    linestyle=:dash,
    color="black"
)

# Plot erorr bars for fitness comparison between replicates in hierarchical
# model
errorbars!(
    ax[2],
    df_fitness[(df_fitness.rep.=="R1"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :mean_single],
    df_fitness[(df_fitness.rep.=="R1"), :std_single],
    direction=:x,
    color=(:gray, 0.5)
)
errorbars!(
    ax[2],
    df_fitness[(df_fitness.rep.=="R1"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :std_single],
    direction=:y,
    color=(:gray, 0.25)
)

# Plot mean for fitness comparison between replicates in hierarchical model
scatter!(
    ax[2],
    df_fitness[(df_fitness.rep.=="R1"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :mean_single],
    markersize=5,
    color=ColorSchemes.seaborn_colorblind[2]
)

ax[2].title = "single dataset model"
ax[2].xlabel = "replicate R1 fitness"
ax[2].ylabel = "replicate R2 fitness"

# -----------------------------------------------------------------------------

# Add identity line
lines!(
    ax[3],
    repeat(
        [[
            minimum(df_fitness.mean) * 1.1,
            maximum(df_fitness.mean) * 1.1,
        ]],
        2
    )...,
    linestyle=:dash,
    color="black"
)

# Extract data
data = df_fitness[df_fitness.rep.=="R1", :]

# Plot hierarchical hyperfitness vs ground truth
errorbars!(
    ax[3],
    data.hyperfitness,
    data.mean_h,
    data.std_h,
    color=(:gray, 0.25)
)
scatter!(
    ax[3],
    data.hyperfitness,
    data.mean_h,
    label="hierarchical hyperfitness",
    markersize=5,
)

# Plot single-dataset model vs ground turth
scatter!(
    ax[3],
    data.hyperfitness,
    data.mean_single_mean,
    label="⟨single dataset fitness⟩",
    markersize=5
)

# Add legend
axislegend(ax[3], position=:lt, labelsize=13, framevisible=false)

# Label axis
ax[3].xlabel = "ground truth hyperfitness"
ax[3].ylabel = "inferred parameter"

# -----------------------------------------------------------------------------

# Extract data
data = df_fitness[df_fitness.rep.=="R1", :]

# Plot ECDF for hierarchical model
ecdfplot!(
    ax[4],
    abs.(data.mean_h .- data.hyperfitness),
    label="hierarchical model"
)

# Plot ECDF for single-dataset model
ecdfplot!(
    ax[4],
    abs.(data.mean_single_mean .- data.hyperfitness),
    label="single dataset model"
)

# Add legend
axislegend(ax[4], position=:rb)

# Label axis
ax[4].xlabel = "|mean - ground truth hyperfitness|"
ax[4].ylabel = "ECDF"

save("./output/figs/advi_hierarchical_vs_single.pdf", fig)
fig