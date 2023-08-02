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
file = first(Glob.glob("./output/advi_meanfield*"))

# Load distribution
advi_results = JLD2.load(file)
ids_advi = advi_results["ids"]
dist_advi = advi_results["dist"]
var_advi = advi_results["var"]

# Extract distribution parameters
dist_params = hcat(Distributions.params(dist_advi)...)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples
n_samples = 10_000

# Sample from ADVI joint distribution and convert to dataframe
df_samples = DF.DataFrame(Random.rand(dist_advi, n_samples)', var_advi)

# Locate hyperparameter variables
θ_idx = occursin.("θ̲⁽ᵐ⁾", String.(var_advi))
# Find columns with fitness parameter deviation 
τ_idx = occursin.("τ̲⁽ᵐ⁾", String.(var_advi))
θ_tilde_idx = occursin.("θ̲̃⁽ᵐ⁾", String.(var_advi))

# Compute samples for individual barcode fitness values. These are not
# parameters included in the joint distribution, but can be computed from
# samples of the other parameters
df_samples = hcat(
    df_samples,
    DF.DataFrame(
        Matrix(df_samples[:, θ_idx])[:, geno_idx] .+
        (
            exp.(Matrix(df_samples[:, τ_idx])) .*
            Matrix(df_samples[:, θ_tilde_idx])
        ),
        Symbol.("s_" .* ids_advi)
    )
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
    color=:gray,
    alpha=0.5,
    linewidth=2
)

# Save figure into pdf
save("./output/figs/advi_logfreqratio_ppc_neutral.pdf", fig)
save("./output/figs/advi_logfreqratio_ppc_neutral.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute summary statistics for each barcode
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize dataframe to save individual barcode summary statistics
df_bc = DF.DataFrame()

# Find barcode variables
s_names = names(df_samples)[occursin.("s_", names(df_samples))]
# Find genotype variables
θ_names = names(df_samples)[occursin.("θ̲⁽ᵐ⁾", names(df_samples))]

# Define percentiles to include
per = [2.5, 97.5, 16, 84]

# Loop through barcodes
for (i, bc) in enumerate(s_names)
    # Extract fitness chain
    fitness_s = @view df_samples[:, bc]
    # Extract hyperparameter chain
    fitness_θ = @view df_samples[:, θ_names[geno_idx[i]]]

    # Compute summary statistics
    fitness_summary = Dict(
        :s_mean => StatsBase.mean(fitness_s),
        :s_median => StatsBase.median(fitness_s),
        :s_std => StatsBase.std(fitness_s),
        :s_var => StatsBase.var(fitness_s),
        :s_skewness => StatsBase.skewness(fitness_s),
        :s_kurtosis => StatsBase.kurtosis(fitness_s),
        :θ_mean => StatsBase.mean(fitness_θ),
        :θ_median => StatsBase.median(fitness_θ),
        :θ_std => StatsBase.std(fitness_θ),
        :θ_var => StatsBase.var(fitness_θ),
        :θ_skewness => StatsBase.skewness(fitness_θ),
        :θ_kurtosis => StatsBase.kurtosis(fitness_θ),)

    # Loop through percentiles
    for p in per
        setindex!(
            fitness_summary,
            StatsBase.percentile(fitness_s, p),
            Symbol("s_$(p)_percentile")
        )
        setindex!(
            fitness_summary,
            StatsBase.percentile(fitness_θ, p),
            Symbol("θ_$(p)_percentile")
        )
    end # for

    # Convert to dataframe
    df = DF.DataFrame(fitness_summary)
    # Add barcode
    df[!, :id] .= bc
    # Append to dataframe
    DF.append!(df_bc, df)
end # for

# Add barcode and genotype
DF.insertcols!(
    df_bc,
    :barcode => [split(x, "_")[2] for x in df_bc.id],
    :genotype => geno_unique[geno_idx]
)

# Sort by barcode
DF.sort!(df_bc, :barcode)

# Append fitness and growth rate value
DF.leftjoin!(
    df_bc,
    unique(
        data[.!(data.neutral),
            [:barcode, :genotype, :fitness, :growth_rate]]
    );
    on=[:barcode, :genotype]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare fitness of individal barcodes with ground truth
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350 * 2, 350))

# Add axis
ax = [Axis(fig[1, i]) for i in 1:2]

# Add identity line
lines!(ax[1], repeat([[-0.05, 1.25]], 2)..., linestyle=:dash, color="black")

# Error bars
errorbars!(
    ax[1],
    df_bc.fitness,
    df_bc.s_median,
    abs.(df_bc.s_median - df_bc[:, "s_16.0_percentile"]),
    abs.(df_bc.s_median - df_bc[:, "s_84.0_percentile"]),
    color=(:gray, 0.5),
    direction=:y,
)

# Add points
scatter!(
    ax[1],
    df_bc.fitness,
    df_bc.s_mean,
    markersize=5
)

# Label axis
ax[1].xlabel = "true barcode fitness"
ax[1].ylabel = "inferred fitness"

df_genotype = DF.combine(
    DF.groupby(df_bc, :genotype),
    :θ_mean => StatsBase.mean,
    :θ_mean => StatsBase.median,
    Symbol("θ_16.0_percentile") => StatsBase.mean,
    Symbol("θ_84.0_percentile") => StatsBase.mean,
    :fitness => StatsBase.mean
)

# Rename columns
DF.rename!(
    df_genotype,
    :θ_mean_mean => :θ_mean,
    :θ_mean_median => :θ_median,
    Symbol("θ_16.0_percentile_mean") => :θ_16,
    Symbol("θ_84.0_percentile_mean") => :θ_84,
    :fitness_mean => :fitness
)

# Add identity line
lines!(ax[2], repeat([[-0.05, 1.25]], 2)..., linestyle=:dash, color="black")

# Error bars
errorbars!(
    ax[2],
    df_genotype.fitness,
    df_genotype.θ_mean,
    abs.(df_genotype.θ_mean .- df_genotype.θ_16),
    abs.(df_genotype.θ_mean .- df_genotype.θ_84),
    color=(:gray, 0.5),
    direction=:y,
)

# Add points
scatter!(
    ax[2],
    df_genotype.fitness,
    df_genotype.θ_mean,
    markersize=5
)

# Label axis
ax[2].xlabel = "true genotype fitness"
ax[2].ylabel = "inferred fitness"

# Save figure 
save("./output/figs/advi_vs_true_fitness.pdf", fig)
save("./output/figs/advi_vs_true_fitness.svg", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(300 * 2, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|barcode median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(df_bc.s_median .- df_bc.fitness))
# Plot ECDF
ax2 = Axis(fig[1, 2], xlabel="|genotype median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax2, abs.(df_genotype.θ_median .- df_genotype.fitness))


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
bc_plot = StatsBase.sample(1:length(s_names), n_row * n_col)

# find standard deviation variables
σ_names = names(df_samples)[occursin.("logσ̲⁽ᵐ⁾", names(df_samples))]

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
            data[
                data.barcode.==replace(s_names[bc_plot[counter]], "s_" => ""),
                :],
            :time
        )

        # Define colors
        local colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :mutant_mean_fitness => Symbol(s_names[bc_plot[counter]]),
            :mutant_std_fitness => Symbol(σ_names[bc_plot[counter]]),
            :population_mean_fitness => :s̲ₜ,
        )
        # Compute posterior predictive checks
        local ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
            df_samples, n_ppc; model=:normal, param=param
        )
        # Plot posterior predictive checks
        BayesFitUtils.viz.ppc_time_series!(
            ax, qs, ppc_mat; colors=colors
        )

        # Add scatter of data
        scatterlines!(ax, diff(log.(data_bc.freq)), color=:black, linewidth=2.5)

        # Add title
        ax.title = "$(replace(s_names[bc_plot[counter]], "s_" => ""))"
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

fig

save("./output/figs/advi_logfreqratio_ppc_mutant.pdf", fig)
save("./output/figs/advi_logfreqratio_ppc_mutant.svg", fig)