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
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_001/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read chain into memory 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading MCMC Chains...")

# Define file
file = first(Glob.glob("./output/chain_joint_fitness_3000*"))

# Load chain
ids, chn = values(JLD2.load(file))

# Convert chain to tidy dataframe
df_chn = DF.DataFrame(chn)

# Remove the string "mut" from mutant names
bc_num = replace.(ids, "mut" => "")

# Find columns with mutant fitness values and error
s_names = MCMCChains.namesingroup(chn, :s̲⁽ᵐ⁾)
σ_names = MCMCChains.namesingroup(chn, :σ̲⁽ᵐ⁾)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot Trace and density plots for population mean fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting trace and densities for population parameters...")
# Extract variable names
var_names = vcat(
    [MCMCChains.namesingroup(chn, :s̲ₜ), MCMCChains.namesingroup(chn, :logσ̲ₜ)]...
)

# Initialize figure
fig = Figure(resolution=(600, 800))

# Generate mcmc_trace_density! plot
BayesFitUtils.viz.mcmc_trace_density!(fig, chn[var_names]; alpha=0.5)

# Save figure 
save("./output/figs/mcmc_trace_density_popmeanfitness.pdf", fig)
save("./output/figs/mcmc_trace_density_popmeanfitness.svg", fig)

fig
##

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
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1.0, length(qs)))

# Compute posterior predictive checks
ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
    chn, n_ppc; param=param, model=:normal
)

# Define time
t = vec(collect(axes(ppc_mat, 2)) .+ 1)

# Initialize figure
fig = Figure(resolution=(450, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="ln(fₜ₊₁/fₜ)",
    title="neutral lineages PPC"
)

# Plot posterior predictive checks
BayesFitUtils.viz.ppc_time_series!(
    ax, qs, ppc_mat; colors=colors, time=t
)

# Plot log-frequency ratio of neutrals
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    data[data.neutral, :];
    freq_col=:freq,
    color=:black,
    alpha=1.0,
    linewidth=2
)

# Save figure into pdf
save("./output/figs/mcmc_logfreqratio_ppc_neutral_posterior.pdf", fig)
save("./output/figs/mcmc_logfreqratio_ppc_neutral_posterior.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Trace and density plots for example mutants
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting trace and densities for a few barcodes...")
# Define barcodes to include
var_names = StatsBase.sample(s_names, 8)

# Initialize figure
fig = Figure(resolution=(600, 800))

# Generate mcmc_trace_density! plot
BayesFitUtils.viz.mcmc_trace_density!(fig, chn[var_names]; alpha=0.5)

# Save figure 
save("./output/figs/mcmc_trace_density_mutants.pdf", fig)
save("./output/figs/mcmc_trace_density_mutants.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute summary statistics for fitness values
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Computing summary statistics per chain...")
# Define percentiles to include
per = [2.5, 97.5, 16, 84]

# Convert chain to tidy dataframe
df = DF.stack(DF.DataFrame(chn[s_names]), DF.Not([:iteration, :chain]))

# FILTER 95% #

# Group data by barcode
df_group = DF.groupby(df, :variable)

# Initialize dataframe to save summaries
df_filt = DF.DataFrame()

# Loop through groups
for g in df_group
    # Filter data
    DF.append!(
        df_filt,
        g[(g.value.≥StatsBase.percentile(g.value, 5)).&(g.value.≤StatsBase.percentile(g.value, 95)),
            :]
    )
end # group

# Compute summary statistics
df_summary = DF.combine(
    DF.groupby(df_filt, :variable),
    :value => StatsBase.median,
    :value => StatsBase.mean,
    :value => StatsBase.std,
    :value => StatsBase.var,
    :value => StatsBase.skewness,
    :value => StatsBase.kurtosis,
)

# Add barcode column to match
df_summary[!, :barcode] = [
    Dict(unique(df_summary.variable) .=> ids)[x] for x in df_summary.variable
]

# Loop through percentiles
for p in per
    # Compute and add percentile
    DF.leftjoin!(
        df_summary,
        # Rename column from :value_function to :p_percentile
        DF.rename!(
            # Compute percentile p for each group
            DF.combine(
                # Group MCMC chains by :variable, :value
                DF.groupby(df_filt[:, [:variable, :value]], :variable),
                # Define anonymous function to compute percentile
                :value => x -> StatsBase.percentile(x, p)
            ),
            :value_function => Symbol("$(p)_percentile")
        );
        on=:variable
    )
end # for

# Rename columns
DF.rename!(
    df_summary,
    :value_median => :median,
    :value_mean => :mean,
    :value_std => :std,
    :value_var => :var,
    :value_skewness => :skewness,
    :value_kurtosis => :excess_kurtosis,
)

# Append fitness and growth rate value
DF.leftjoin!(
    df_summary,
    unique(data[.!(data.neutral), [:barcode, :fitness, :growth_rate]]);
    on=:barcode
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting comparison between inferred fitness and ground truth")
# Initialize figure
fig = Figure(resolution=(300, 300))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="true fitness value",
    ylabel="inferred fitness",
)

# Plot identity line
lines!(
    ax,
    repeat([[minimum(df_summary.median), maximum(df_summary.median)]], 2)...;
    color=:black, linestyle="--"
)

# Add x-axis error bars
errorbars!(
    ax,
    df_summary.fitness,
    df_summary.median,
    abs.(df_summary.median .- df_summary[:, "16.0_percentile"]),
    abs.(df_summary.median .- df_summary[:, "84.0_percentile"]),
    color=(:gray, 0.5),
    direction=:y
)

# Plot comparison
scatter!(ax, df_summary.fitness, df_summary.median, markersize=8)

save("./output/figs/mcmc_fitness_comparison.pdf", fig)
save("./output/figs/mcmc_fitness_comparison.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting diagnostic ECDFs...")
# Initialize figure
fig = Figure(resolution=(350, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(df_summary.median .- df_summary.fitness))

save("./output/figs/mcmc_median_true_ecdf.pdf", fig)
save("./output/figs/mcmc_median_true_ecdf.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF for true fitness z-score 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Fit Gaussian distribution to chains
dist_fit = [Distributions.fit(Distributions.Normal, chn[x]) for x in s_names]
# Extract parameters
dist_param = hcat(
    first.(Distributions.params.(dist_fit)),
    last.(Distributions.params.(dist_fit))
)
# Compute Z-score of true fitness values
fitness_zscore = (df_summary.fitness .- dist_param[:, 1]) ./ dist_param[:, 2]
# Initialize figure
fig = Figure(resolution=(350, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|z-score|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(fitness_zscore))

save("./output/figs/mcmc_zscore_ecdf.pdf", fig)
save("./output/figs/mcmc_zscore_ecdf.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot example posterior predictive checks
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting posterior predictive checks for example mutants...")
# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [4, 4]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# List example barcodes to plot
bc_plot = StatsBase.sample(eachrow(df_summary), n_row * n_col)

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add axis
        local ax = Axis(fig[row, col])

        # Extract data
        data_bc = DF.sort(
            data[data.barcode.==bc_plot[counter].barcode, :], :time
        )

        # Define colors
        local colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :mutant_mean_fitness => Symbol(bc_plot[counter].variable),
            :mutant_std_fitness => Symbol(
                replace(bc_plot[counter].variable, "s" => "logσ")
            ),
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        local ppc_mat = BarBay.stats.logfreq_ratio_mutant_ppc(
            chn, n_ppc; param=param
        )
        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors
        )

        # Add scatter of data
        scatterlines!(ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.5)

        # Add title
        ax.title = "barcode $(first(data_bc.barcode))"
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

save("./output/figs/mcmc_logfreqratio_ppc_mutant.pdf", fig)
save("./output/figs/mcmc_logfreqratio_ppc_mutant.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results...")
# Define file
file = first(Glob.glob("./output/advi_meanfield*3000*csv"))

# Convert results to tidy dataframe
df_advi = CSV.read(file, DF.DataFrame)

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
# Compare inferred population mean fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting comparison between MCMC and ADVI results...")
# Locate variable names in MCMC
mcmc_names = names(df_chn)[occursin.("s̲ₜ", names(df_chn))]
# Locate variables in ADVI
advi_idx = occursin.("s̲ₜ", String.(df_advi.varname))

# Compute mcmc mean and std
mcmc_mean = StatsBase.mean.(eachcol(df_chn[:, mcmc_names]))
mcmc_std = StatsBase.std.(eachcol(df_chn[:, mcmc_names]))

# Extract advi mean and std
mean = df_advi[advi_idx, :mean]
std = df_advi[advi_idx, :std]


# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="MCMC inference",
    ylabel="ADVI inference",
    title="population mean fitness",
    aspect=AxisAspect(1),
)

# Add identity line
lines!(ax, repeat([[0.5, 1.1]], 2)..., linestyle=:dash, color=:black)


# add errorbars
errorbars!(
    mcmc_mean,
    mean,
    mcmc_std,
    color=(:gray, 0.5),
    direction=:x
)
# add errorbars
errorbars!(
    mcmc_mean,
    mean,
    std,
    color=(:gray, 0.5),
    direction=:y
)

# Add points
scatter!(
    mcmc_mean,
    mean,
    markersize=8
)


save("./output/figs/advi_vs_mcmc_popmean.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare inferred mutant relative fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Locate variable names in MCMC
mcmc_names = names(df_chn)[occursin.("s̲⁽ᵐ⁾", names(df_chn))]
# Locate variables in ADVI
advi_idx = occursin.("s̲⁽ᵐ⁾", String.(df_advi.varname))

# Compute mcmc mean and std
mcmc_mean = StatsBase.mean.(eachcol(df_chn[:, mcmc_names]))
mcmc_std = StatsBase.std.(eachcol(df_chn[:, mcmc_names]))

# Extract advi mean and std
mean = df_advi[advi_idx, :mean]
std = df_advi[advi_idx, :std]


# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="MCMC inference",
    ylabel="ADVI inference",
    title="mutant relative fitness",
    aspect=AxisAspect(1),
)

# Add identity line
lines!(ax, repeat([[-0.5, 1.75]], 2)..., linestyle=:dash, color=:black)

# add errorbars
errorbars!(
    mcmc_mean,
    mean,
    mcmc_std,
    color=(:gray, 0.5),
    direction=:x
)
# add errorbars
errorbars!(
    mcmc_mean,
    mean,
    std,
    color=(:gray, 0.5),
    direction=:y
)

# Add points
scatter!(
    mcmc_mean,
    mean,
    markersize=8
)

save("./output/figs/advi_vs_mcmc_mutant.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting comparison between ADVI results and ground truth...")
# Extract dataframe with unique pairs of barcodes and fitness values
data_fitness = DF.sort(
    unique(data[(.!data.neutral), [:barcode, :fitness]]), :barcode
)

# Extract ADVI inferred fitness
advi_fitness = DF.sort(
    df_advi[(df_advi.vartype.=="bc_fitness"), [:id, :mean, :std]],
    :id
)

# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="true fitness value",
    ylabel="ADVI inferred fitness",
)

# Plot identity line
lines!(
    ax,
    repeat(
        [[-0.1, 1.75]], 2
    )...;
    color=:black,
    linestyle="--"
)

# Add x-axis error bars
errorbars!(
    ax,
    data_fitness.fitness,
    advi_fitness.mean,
    advi_fitness.std,
    color=(:gray, 0.5),
    direction=:y
)

# Plot comparison
scatter!(
    ax,
    data_fitness.fitness,
    advi_fitness.mean,
    markersize=8
)

save("./output/figs/advi_fitness_true.pdf", fig)

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
ppc_color = get(ColorSchemes.Blues_9, LinRange(0.25, 1.0, length(qs)))

# Initialize figure
fig = Figure(resolution=(400, 350))

# Compute posterior predictive checks
ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
    df_samples, n_ppc; model=:normal, param=param
)

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="time point",
    ylabel="ln(fₜ₊₁/fₜ)",
    title="neutral lineages PPC",
)

# Plot posterior predictive checks
BayesFitUtils.viz.ppc_time_series!(
    ax, qs, ppc_mat; colors=ppc_color, time=sort(unique(data.time))[2:end]
)

# Plot log-frequency ratio of neutrals
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    data[data.neutral, :];
    freq_col=:freq,
    color=:black,
    linewidth=2
)

# Save figure into pdf
save("./output/figs/advi_logfreqratio_ppc_neutral.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for barcodes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting posterior predictive checks for example barcodes...")
# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [4, 4]

# List example barcodes to plot
bc_plot = StatsBase.sample(
    unique(data[.!data.neutral, :barcode]), n_row * n_col
)

# Extract unique mutant/fitnes variable name pairs
bc_var = df_advi[(df_advi.vartype.=="bc_fitness"), [:id, :varname]]

# Generate dictionary from mutant name to fitness value
bc_var_dict = Dict(zip(bc_var.id, bc_var.varname))

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs)))

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
        ax = Axis(gl[1, 1:6])

        # Extract data
        data_bc = DF.sort(
            data[(data.barcode.==bc_plot[counter]), :], :time
        )

        # Extract variables for barcode PPC
        global vars_bc = [
            names(df_samples)[occursin.("s̲ₜ", names(df_samples))]
            bc_var_dict[bc_plot[counter]]
            replace(bc_var_dict[bc_plot[counter]], "s" => "logσ")
        ]


        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :mutant_mean_fitness => Symbol(bc_var_dict[bc_plot[counter]]),
            :mutant_std_fitness => Symbol(
                replace(bc_var_dict[bc_plot[counter]], "s" => "logσ")
            ),
            :population_mean_fitness => Symbol("s̲ₜ"),
        )
        # Compute posterior predictive checks
        local ppc_mat = BarBay.stats.logfreq_ratio_mutant_ppc(
            df_samples[:, Symbol.(vars_bc)],
            n_ppc;
            model=:normal,
            param=param
        )

        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax,
            qs,
            ppc_mat;
            colors=colors,
            time=sort(unique(data.time))[2:end]
        )

        # Plot log-frequency ratio of neutrals
        BayesFitUtils.viz.logfreq_ratio_time_series!(
            ax,
            data_bc,
            freq_col=:freq,
            color=:black,
            linewidth=3,
            markersize=12
        )

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

println("Done!")