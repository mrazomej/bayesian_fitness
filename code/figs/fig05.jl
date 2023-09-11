##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

import Revise
# Import project package
import BayesFitUtils

# Import library to perform Bayesian inference
import BayesFitness

# Import basic math
import Random
import Distributions
import LinearAlgebra
import StatsBase

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes
import ColorTypes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

Random.seed!(18)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
df_counts = CSV.read(
    "$(git_root())/data/logistic_growth/data_007/tidy_data.csv", DF.DataFrame
)

# Generate a barcode to genotype dictionary
bc_to_geno = Dict(
    zip(collect(eachcol(unique(df_counts[:, [:barcode, :genotype]])))...)
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load ADVI results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results...")

# Define file
file = "$(git_root())/code/processing/" *
       "data007_logistic_hierarchicalgenotype_1000bc_01env_01rep/" *
       "output/advi_meanfield_hierarchicalgenotypes_01samples_10000steps.csv"

# Convert results to tidy dataframe
df_advi = CSV.read(file, DF.DataFrame)

# Extract bc fitness values
df_fitness = df_advi[(df_advi.vartype.=="bc_fitness"), :]

# Add genotype information
df_fitness[!, :genotype] = [bc_to_geno[x] for x in df_fitness.id]

# Extract and append hyperfitness values for each fitness value
DF.leftjoin!(
    df_fitness,
    DF.rename(
        df_advi[(df_advi.vartype.=="bc_hyperfitness"),
            [:mean, :std, :varname, :id]],
        :mean => :mean_h,
        :std => :std_h,
        :varname => :varname_h,
        :id => :genotype,
    );
    on=:genotype
)

# Extract and append ground truth fitness values
DF.leftjoin!(
    df_fitness,
    DF.rename(
        unique(df_counts[.!(df_counts.neutral), [:barcode, :fitness]]),
        :barcode => :id
    );
    on=:id
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
# Load ADVI results for other two cases
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = "$(git_root())/code/processing/" *
       "data007_logistic_hierarchicalgenotype_1000bc_01env_01rep/" *
       "output/advi_meanfield_01samples_10000steps.csv"

# Convert results to tidy dataframe
df_advi_single = CSV.read(file, DF.DataFrame)

# Define file
file = "$(git_root())/code/processing/" *
       "data007_logistic_hierarchicalgenotype_1000bc_01env_01rep/" *
       "output/advi_meanfield_genotype-group_01samples_10000steps.csv"

# Convert results to tidy dataframe
df_advi_group = CSV.read(file, DF.DataFrame)

# Extract bc fitness values
df_fitness_single = df_advi_single[(df_advi_single.vartype.=="bc_fitness"), :]
df_fitness_group = df_advi_group[(df_advi_group.vartype.=="bc_fitness"), :]

# Add genotype information
df_fitness_single[!, :genotype] = [bc_to_geno[x] for x in df_fitness_single.id]
# Rename column to genotype
DF.rename!(df_fitness_group, :id => :genotype)

# Extract and append ground truth fitness values
DF.leftjoin!(
    df_fitness_single,
    DF.rename(
        unique(df_counts[.!(df_counts.neutral), [:barcode, :fitness]]),
        :barcode => :id
    );
    on=:id
)

DF.leftjoin!(
    df_fitness_group,
    DF.rename(
        DF.combine(
            DF.groupby(df_fitness, :genotype), :fitness => StatsBase.mean
        ),
        :fitness_mean => :fitness
    );
    on=:genotype
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Set figure layout
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(1200, 700))

# Add grid layout for illustrator diagram
gl_illustrator = fig[1, 1] = GridLayout()

# Add grid lauout for datasets
gl_data = fig[2, 1] = GridLayout()

# Add grid layout for true hyperfitness vs true fitness
gl_hyper_vs_fit_true = fig[1, 2:4] = GridLayout()

# Add grid layout for ECDF plots
gl_ecdf = fig[2, 2] = GridLayout()

# Add grid layout for posterior predictive checks
gl_ppc = fig[2, 3:4] = GridLayout()

# ---------------------------------------------------------------------------- #

# Add axis
ax = Axis(
    gl_data[1, 1],
    xlabel="time",
    yscale=log10,
    aspect=AxisAspect(1)
)

# Group mutant barcodes by genotype
df_group = DF.groupby(df_counts[.!df_counts.neutral, :], :genotype)

# Define colors for genotypes
colors = ColorSchemes.glasbey_hv_n256

# Loop through genotypes
for (i, data) in enumerate(df_group)
    # Plot mutant barcode trajectories
    BayesFitUtils.viz.bc_time_series!(
        ax,
        data,
        quant_col=:freq,
        zero_lim=1E-10,
        alpha=0.3,
        zero_label="extinct",
        color=colors[i]
    )
end # for

# Plot neutral barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax,
    df_counts[df_counts.neutral, :],
    quant_col=:freq,
    zero_lim=1E-10,
    color=ColorSchemes.Blues_9[end],
)

# ----------------------------------------------------------------------------

# Add axis for true vs inferred hyperfitness 
ax = Axis(
    gl_hyper_vs_fit_true[1, 1],
    xlabel="ground truth genotype fitness",
    ylabel="inferred hyper-fitness",
    aspect=AxisAspect(1),
    title="hierarchical model",
)

# Add identity line
lines!(
    ax,
    repeat(
        [[minimum(df_fitness.mean_h), maximum(df_fitness.mean_h)] .* 1.05], 2
    )...,
    linestyle=:dash,
    color=:black
)

# Plot errorbars
errorbars!(
    ax,
    df_fitness.fitness,
    df_fitness.mean_h,
    df_fitness.std_h,
    color=(:gray, 0.5)
)

# Plot mean point
scatter!(
    ax,
    df_fitness.fitness,
    df_fitness.mean_h,
    markersize=5,
    color=(ColorSchemes.seaborn_colorblind[1], 0.75)
)

# ----------------------

# Add axis for true vs inferred hyperfitness 
ax = Axis(
    gl_hyper_vs_fit_true[1, 2],
    xlabel="ground truth barcode fitness",
    ylabel="inferred barcode fitness",
    aspect=AxisAspect(1),
    title="single-barcode inference"
)

# Add identity line
lines!(
    ax,
    repeat(
        [[minimum(df_fitness.mean_h), maximum(df_fitness.mean_h)] .* 1.05], 2
    )...,
    linestyle=:dash,
    color=:black
)

# Plot errorbars
errorbars!(
    ax,
    df_fitness_single.fitness,
    df_fitness_single.mean,
    df_fitness_single.std,
    color=(:gray, 0.5)
)

# Plot mean point
scatter!(
    ax,
    df_fitness_single.fitness,
    df_fitness_single.mean,
    markersize=5,
    color=(ColorSchemes.seaborn_colorblind[2], 0.75)
)

# ----------------------

# Add axis for true vs inferred hyperfitness 
ax = Axis(
    gl_hyper_vs_fit_true[1, 3],
    xlabel="ground truth genotype fitness",
    ylabel="inferred genotype fitness",
    aspect=AxisAspect(1),
    title="pooled-barcodes inference"
)

# Add identity line
lines!(
    ax,
    repeat(
        [[minimum(df_fitness.mean_h), maximum(df_fitness.mean_h)] .* 1.05], 2
    )...,
    linestyle=:dash,
    color=:black
)

# Plot errorbars
errorbars!(
    ax,
    df_fitness_group.fitness,
    df_fitness_group.mean,
    df_fitness_group.std,
    color=(:gray, 0.5)
)

# Plot mean point
scatter!(
    ax,
    df_fitness_group.fitness,
    df_fitness_group.mean,
    markersize=5,
    color=(ColorSchemes.seaborn_colorblind[3], 0.75)
)

# ----------------------------------------------------------------------------

# Add axis for ECDF
ax = Axis(
    gl_ecdf[1, 1],
    xlabel="|mean - ground truth fitness|",
    ylabel="ECDF",
    aspect=AxisAspect(1)
)

# Plot hierarchical model ECDF
ecdfplot!(
    ax,
    abs.(df_fitness.mean_h .- df_fitness.fitness),
    color=ColorSchemes.seaborn_colorblind[1],
    label="hierarchical model",
    linewidth=2.5,
)

# Plot single-barcode inference ECDF
ecdfplot!(
    ax,
    abs.(df_fitness_single.mean .- df_fitness_single.fitness),
    color=ColorSchemes.seaborn_colorblind[2],
    label="single-barcode",
    linewidth=2.5,
)

# Plot grouped-barcode inference ECDF
ecdfplot!(
    ax,
    abs.(df_fitness_group.mean .- df_fitness_group.fitness),
    color=ColorSchemes.seaborn_colorblind[3],
    label="pooled-barcodes",
    linewidth=2.5,
)

axislegend(ax, position=:rb, framevisible=false)

# ---------------------------------------------------------------------------- #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [2, 3]

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
            local ppc_mat = BayesFitness.stats.logfreq_ratio_popmean_ppc(
                df_samples, n_ppc; model=:normal, param=param
            )

            # Define time
            t = vec(collect(axes(ppc_mat, 2)) .+ 1)

            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax, qs, ppc_mat; colors=colors, time=t
            )

            # Plot log-frequency ratio of neutrals
            BayesFitUtils.viz.logfreq_ratio_time_series!(
                ax,
                df_counts[df_counts.neutral, :];
                freq_col=:freq,
                color=:black,
                alpha=1.0,
                linewidth=1.5
            )

            # Hide axis decorations
            hidedecorations!.(ax, grid=false)

            ax.title = "neutral lineages"
            # ax.titlesize = 18

            counter += 1

            continue
        end # if

        # Extract data
        data_bc = DF.sort(
            df_counts[df_counts.barcode.==bc_plot[counter].id, :], :time
        )

        # Define colors
        local colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :bc_mean_fitness => Symbol(bc_plot[counter].varname),
            :bc_std_fitness => Symbol(
                replace(bc_plot[counter].varname, "s" => "logσ")
            ),
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        local ppc_mat = BayesFitness.stats.logfreq_ratio_bc_ppc(
            df_samples, n_ppc; model=:normal, param=param
        )
        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors
        )

        # Add scatter of data
        scatterlines!(ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.0)

        # Define fitness ranges to display in title
        vals = [
            round(bc_plot[counter].mean; sigdigits=2),
            round(bc_plot[counter].std; sigdigits=2),
        ]

        # Add title
        ax.title = "s⁽ᵐ⁾= $(vals[1])±$(vals[2])"

        ## == Plot format == ##

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(gl_ppc[end, :, Bottom()], "time points", fontsize=18)
# Add y-axis label
Label(gl_ppc[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=18)
# # Set spacing
rowgap!(gl_ppc, 0)
colgap!(gl_ppc, 4)

# ---------------------------------------------------------------------------- #

# Add subplot labels
Label(
    gl_illustrator[1, 1, TopLeft()], "(A)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_data[1, 1, TopLeft()], "(B)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_hyper_vs_fit_true[1, 1, TopLeft()], "(C)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_hyper_vs_fit_true[1, 2, TopLeft()], "(D)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_hyper_vs_fit_true[1, 3, TopLeft()], "(E)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_ecdf[1, 1, TopLeft()], "(F)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add subplot labels
Label(
    gl_ppc[1, 1, TopLeft()], "(G)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

save("$(git_root())/doc/figs/fig05B-G.pdf", fig)

fig