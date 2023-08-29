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
import PDFmerger

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
df = CSV.read(
    "$(git_root())/data/aguilar_2023/tidy_counts_oligo.csv", DF.DataFrame
)

# Remove repeat "N/A"
df = df[df.rep.!="N/A", :]

# Select R1 as representative dataset
data = df[df.rep.=="R1", :]

# Generate dictionary from mutants to genotypes
oligo_edit_dict = Dict(values.(keys(DF.groupby(data, [:oligo, :edit]))))

# Extract list of mutants as they were used in the inference
oligo_ids = BayesFitness.utils.data_to_arrays(data; id_col=:oligo)[:mut_ids]

# Extract genotypes in the order they were used in the inference
edits = [oligo_edit_dict[m] for m in oligo_ids]

# Find unique genotypes
edits_unique = unique(edits)
# Define number of unique genotypes
n_edits = length(edits_unique)
# Define genotype indexes
edits_idx = indexin(edits, edits_unique)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inferences for population mean fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
files = Glob.glob("./output/advi_meanfield_pop*50*")

# Initialize dictionary to save outputs
advi_output = Dict()

# Loop through files
for (i, file) in enumerate(files)
    # Load distribution
    setindex!(advi_output, JLD2.load(file), "R$(i)")
end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples
n_samples = 10_000

# Loop through each replicate
for rep in eachindex(advi_output)

    # Extract variables
    ids_advi = advi_output[rep]["ids"]
    dist_advi = advi_output[rep]["dist"]
    var_advi = advi_output[rep]["var"]

    # Sample from ADVI joint distribution and convert to dataframe
    df_samples = DF.DataFrame(Random.rand(dist_advi, n_samples)', var_advi)

    # Append dataframe to dictionary
    setindex!(advi_output[rep], df_samples, "df_samples")
end # for

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
n_ppc = 100

# Define quantiles to compute
qs = [0.05, 0.68, 0.95]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1.0, length(qs)))

rm("./output/figs/advi_logfreqratio_ppc_neutral_prior.pdf"; force=true)
# Loop through repeats
for rep in sort(collect(eachindex(advi_output)))
    println("Generating plot for $(rep)")
    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_popmean_ppc(
        advi_output[rep]["df_samples"], n_ppc; model=:normal, param=param
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
        title="neutral lineages PPC | $(rep)"
    )

    # Plot posterior predictive checks
    BayesFitUtils.viz.ppc_time_series!(
        ax, qs, ppc_mat; colors=colors, time=t
    )

    # Plot log-frequency ratio of neutrals
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax,
        df[(df.neutral).&(df.rep.==rep), :];
        freq_col=:freq,
        id_col=:oligo,
        color=:gray,
        alpha=0.5,
        linewidth=2
    )

    # Save figure into pdf
    save("./output/figs/temp.pdf", fig)

    # Append pdf
    PDFmerger.append_pdf!(
        "./output/figs/advi_logfreqratio_ppc_neutral_prior.pdf",
        "./output/figs/temp.pdf",
        cleanup=true
    )
end # for


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

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
        Matrix(df_samples[:, θ_idx])[:, edits_idx] .+
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
n_ppc = 100

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
    id_col=:oligo,
    color=:gray,
    alpha=0.5,
    linewidth=2
)

# Save figure into pdf
save("./output/figs/advi_logfreqratio_ppc_neutral.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute summary statistics for each barcode
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize dataframe to save individual barcode summary statistics
df_oligo = DF.DataFrame()

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
    fitness_θ = @view df_samples[:, θ_names[edits_idx[i]]]

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
    DF.append!(df_oligo, df)
end # for

# Add barcode and genotype
DF.insertcols!(
    df_oligo,
    :oligo => [split(x, "_")[2] for x in df_oligo.id],
    :edit => edits_unique[edits_idx]
)

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
oligo_plot = StatsBase.sample(1:length(s_names), n_row * n_col)

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
        data_oligo = DF.sort(
            data[
                data.oligo.==replace(s_names[oligo_plot[counter]], "s_" => ""),
                :],
            :time
        )

        # Define colors
        local colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1.0, length(qs)))

        # Define dictionary with corresponding parameters for variables needed
        # for the posterior predictive checks
        local param = Dict(
            :mutant_mean_fitness => Symbol(s_names[oligo_plot[counter]]),
            :mutant_std_fitness => Symbol(σ_names[oligo_plot[counter]]),
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
        scatterlines!(
            ax, diff(log.(data_oligo.freq)), color=:black, linewidth=2.5
        )

        # Add title
        ax.title = "$(replace(s_names[oligo_plot[counter]], "s_" => ""))"
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

save("./output/figs/advi_logfreqratio_ppc_mutant.pdf", fig)
fig
