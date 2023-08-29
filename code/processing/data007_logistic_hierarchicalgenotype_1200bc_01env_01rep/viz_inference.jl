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
    "$(git_root())/data/logistic_growth/data_007/tidy_data.csv", DF.DataFrame
)

# Generate dictionary from mutants to genotypes
mut_geno_dict = Dict(values.(keys(DF.groupby(data, [:barcode, :genotype]))))

# Extract list of mutants as they were used in the inference
mut_ids = BayesFitness.utils.data2arrays(data)[:mut_ids]

# Extract genotypes in the order they were used in the inference
genotypes = [mut_geno_dict[m] for m in mut_ids]

# Find unique genotypes
geno_unique = unique(genotypes)
# Define number of unique genotypes
n_geno = length(geno_unique)
# Define genotype indexes
geno_idx = indexin(genotypes, geno_unique)

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
df_advi = BayesFitness.utils.advi2df(
    dist_advi, var_advi, ids_advi; n_rep=1, genotypes=genotypes
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
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1.0, length(qs)))

# Compute posterior predictive checks
ppc_mat = BayesFitness.stats.logfreq_ratio_popmean_ppc(
    df_samples, n_ppc; model=:normal, param=param
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
    alpha=0.25,
    linewidth=2
)

# Save figure into pdf
save("./output/figs/advi_logfreqratio_ppc_neutral.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Extract dataframe with unique pairs of barcodes and fitness values
data_fitness = DF.sort(
    unique(data[(.!data.neutral), [:barcode, :fitness]]), :barcode
)

# Extract ADVI inferred fitness
advi_fitness = DF.sort(
    df_advi[(df_advi.vartype.=="mut_fitness"), [:id, :mean, :std]],
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
        [[-0.1, 1.35]], 2
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
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Extract dataframe with unique pairs of barcodes and fitness values
data_fitness_mean = DF.sort(
    DF.combine(
        DF.groupby(data[.!data.neutral, :], :genotype),
        :fitness => StatsBase.mean
    ),
    :genotype
)

# Extract ADVI inferred fitness
advi_fitness_hyper = df_advi[
    (df_advi.vartype.=="mut_hyperfitness"), [:varname, :mean, :std]
]

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
        [[-0.1, 1.35]], 2
    )...;
    color=:black,
    linestyle="--"
)

# Add x-axis error bars
errorbars!(
    ax,
    data_fitness_mean.fitness_mean,
    advi_fitness_hyper.mean,
    advi_fitness_hyper.std,
    color=(:gray, 0.5),
    direction=:y
)

# Plot comparison
scatter!(
    ax,
    data_fitness_mean.fitness_mean,
    advi_fitness_hyper.mean,
    markersize=8
)

save("./output/figs/advi_hyperfitness_true.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(300 * 2, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|barcode median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(data_fitness.fitness .- advi_fitness.mean))
# Plot ECDF
ax2 = Axis(fig[1, 2], xlabel="|genotype median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax2, abs.(data_fitness_mean.fitness_mean .- advi_fitness_hyper.mean))

save("./output/figs/advi_median_true_ecdf.pdf", fig)

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

# List example barcodes to plot
bc_plot = StatsBase.sample(
    unique(data[.!data.neutral, :barcode]), n_row * n_col
)

# Extract unique mutant/fitnes variable name pairs
mut_var = df_advi[(df_advi.vartype.=="mut_fitness"), [:id, :varname]]

# Generate dictionary from mutant name to fitness value
mut_var_dict = Dict(zip(mut_var.id, mut_var.varname))

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
            mut_var_dict[bc_plot[counter]]
            replace(mut_var_dict[bc_plot[counter]], "s" => "logσ")
        ]


        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :mutant_mean_fitness => Symbol(mut_var_dict[bc_plot[counter]]),
            :mutant_std_fitness => Symbol(
                replace(mut_var_dict[bc_plot[counter]], "s" => "logσ")
            ),
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