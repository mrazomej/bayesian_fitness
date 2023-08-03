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
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_003/tidy_data.csv", DF.DataFrame
)

# Define list of environments
envs = [1, 1, 2, 3, 1, 2, 3]
# Define unique environments
env_unique = unique(envs)

# Define color for environments to keep consistency
colors = Dict(env_unique .=> ColorSchemes.tableau_10[1:3])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = first(Glob.glob("./output/advi_meanfield*"))

# Load distribution
advi_results = JLD2.load(file)
ids_advi = advi_results["ids"]
dist_advi = advi_results["dist"]
var_advi = advi_results["var"]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples
n_samples = 10_000

# Sample from ADVI joint distribution and convert to dataframe
df_samples = DF.DataFrame(Random.rand(dist_advi, n_samples)', var_advi)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute summary statistics for each barcode
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize dataframe to save individual barcode summary statistics
df_summary = DF.DataFrame()

# Find barcode variables
s_names = names(df_samples)[occursin.("s̲⁽ᵐ⁾", names(df_samples))]

# Define percentiles to include
per = [2.5, 97.5, 16, 84]

# Loop through barcodes
for (i, bc) in enumerate(s_names)
    # Extract fitness chain
    fitness_s = @view df_samples[:, bc]

    # Compute summary statistics
    fitness_summary = Dict(
        :mean => StatsBase.mean(fitness_s),
        :median => StatsBase.median(fitness_s),
        :std => StatsBase.std(fitness_s),
        :var => StatsBase.var(fitness_s),
        :skewness => StatsBase.skewness(fitness_s),
        :kurtosis => StatsBase.kurtosis(fitness_s),
    )

    # Loop through percentiles
    for p in per
        setindex!(
            fitness_summary,
            StatsBase.percentile(fitness_s, p),
            Symbol("$(p)_percentile")
        )
    end # for

    # Convert to dataframe
    df = DF.DataFrame(fitness_summary)
    # Add barcode
    df[!, :id] .= bc
    # Append to dataframe
    DF.append!(df_summary, df)
end # for

# Add barcode and environment information
DF.insertcols!(
    df_summary,
    :barcode => vcat([repeat([x], length(env_unique)) for x in ids_advi]...),
    :env => vcat(repeat([1, 2, 3], length(ids_advi))...)
)

# Sort by barcode
DF.sort!(df_summary, :barcode)

# Append fitness and growth rate value
DF.leftjoin!(
    df_summary,
    unique(
        data[.!(data.neutral),
            [:env, :barcode, :fitness, :growth_rate]]
    );
    on=[:env, :barcode]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting frequency trajectories...\n")
# Initialize figure
fig = Figure(resolution=(450, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="barcode frequency",
    yscale=log10,
    title="frequency trajectories",
    backgroundcolor=:white,
)

# Define time-environment relation
time_env = Matrix(unique(data[:, [:time, :env]]))

# Loop through each time point
for t = 1:size(time_env, 1)
    # Color plot background
    vspan!(
        ax,
        time_env[t, 1] - 1,
        time_env[t, 1],
        color=(colors[time_env[t, 2]], 0.25)
    )
end # for

# Plot mutant barcode frequency trajectories
BayesFitUtils.viz.bc_time_series!(
    ax,
    data[.!data.neutral, :];
    quant_col=:freq,
    zero_lim=1E-7,
    zero_label="extinct",
    alpha=0.25,
    linewidth=2
)

# Plot Neutral barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax,
    data[data.neutral, :];
    quant_col=:freq,
    zero_lim=1E-7,
    color=ColorSchemes.Blues_9[end],
    alpha=0.9,
    linewidth=2
)

# Set axis limits
xlims!(ax, 1, 7)

save("./output/figs/freq_trajectories.pdf")

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot log-freq ratio trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting log-frequency ratio trajectories...\n")
# Initialize figure
fig = Figure(resolution=(450, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="ln(fₜ₊₁/fₜ)",
    title="log-frequency ratio trajectories",
    backgroundcolor=:white,
)

# Define time-environment relation
time_env = Matrix(unique(data[:, [:time, :env]]))

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

# Plot mutant barcode frequency trajectories
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    data[.!data.neutral, :];
    freq_col=:freq,
    alpha=0.25,
    linewidth=2
)

# Plot Neutral barcode trajectories
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    data[data.neutral, :];
    freq_col=:freq,
    color=ColorSchemes.Blues_9[end],
    alpha=0.9,
    linewidth=2
)

# Set axis limits
xlims!(ax, 1.5, 7.5)

save("./output/figs/logfreqratio_trajectories.pdf")

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
time_env = Matrix(unique(data[:, [:time, :env]]))

# Define time
t = vec(collect(axes(ppc_mat, 2)) .+ 1)

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
xlims!(ax, 1.75, 7.25)

# Save figure into pdf
save("./output/figs/advi_logfreqratio_ppc_neutral.pdf", fig)
save("./output/figs/advi_logfreqratio_ppc_neutral.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare fitness of individal barcodes with ground truth
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(300 * 3, 300))

# Add axis
ax = [
    Axis(
        fig[1, i],
        xlabel="true fitness",
        ylabel="inferred fitness",
        aspect=AxisAspect(1),
    ) for i = 1:3
]

# group data by environment
df_group = DF.groupby(df_summary, :env)
# Extract keys
df_keys = first.(values.(keys(df_group)))

# Loop through environments
for (i, d) in enumerate(df_group)
    # Add identity line
    lines!(
        ax[i],
        repeat([[minimum(d.fitness), maximum(d.fitness)]], 2)...,
        linestyle=:dash,
        color="black"
    )

    # Error bars
    errorbars!(
        ax[i],
        d.fitness,
        d.median,
        abs.(d.median - d[:, "16.0_percentile"]),
        abs.(d.median - d[:, "84.0_percentile"]),
        color=(:gray, 0.5),
        direction=:y,
    )

    # Add points
    scatter!(
        ax[i],
        d.fitness,
        d.mean,
        markersize=7,
        color=colors[df_keys[i]]
    )

    ax[i].title = "env $(df_keys[i])"
end # for

# Save figure 
save("./output/figs/advi_vs_true_fitness.pdf", fig)
save("./output/figs/advi_vs_true_fitness.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350, 350))
# Add axis
ax = Axis(fig[1, 1], xlabel="|median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(
    ax,
    abs.(df_summary.median .- df_summary.fitness),
    color=:black,
    label="all"
)

# group data by environment
df_group = DF.groupby(df_summary, :env)
# Extract keys
df_keys = first.(values.(keys(df_group)))

# Loop through environments
for (i, d) in enumerate(df_group)
    # Plot ECDF
    ecdfplot!(
        ax,
        abs.(d.median .- d.fitness),
        color=colors[df_keys[i]],
        label="env $(df_keys[i])"
    )
end # for

# Add legend
axislegend(ax, position=:rb)

save("./output/figs/advi_median_true_ecdf.pdf", fig)
save("./output/figs/advi_median_true_ecdf.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot example posterior predictive checks
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [4, 4]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# List example barcodes to plot
bc_plot = StatsBase.sample(ids_advi, n_row * n_col)

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
        s_var = df_summary[(df_summary.barcode.==bc_plot[counter]), :id]
        # Extract logσ variable names
        σ_var = replace.(s_var, Ref("s" => "logσ"))
        # Extract mean fitness variables
        sₜ_var = ["s̲ₜ[$(i)]" for i = 1:length(unique(data.time))-1]

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
        time_env = Matrix(unique(data_bc[:, [:time, :env]]))

        # Define time
        t = vec(collect(axes(ppc_mat, 2)) .+ 1)

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
        ax.title = "$(bc_plot[counter])"
        ax.titlesize = 18

        ## == Plot format == ##

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=22)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=22)

# save("./output/figs/advi_logfreqratio_ppc_mutant.pdf", fig)
# save("./output/figs/advi_logfreqratio_ppc_mutant.svg", fig)

fig