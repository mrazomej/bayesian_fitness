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
import PDFmerger

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
df_counts = CSV.read(
    "$(git_root())/data/aguilar_2023/tidy_counts_oligo.csv", DF.DataFrame
)

# Remove repeat "N/A"
df_counts = df_counts[df_counts.rep.!="N/A", :]

# Generate dictionary from mutants to genotypes
oligo_edit_dict = Dict(values.(keys(DF.groupby(df_counts, [:oligo, :edit]))))

# Extract list of mutants as they were used in the inference
oligo_ids = BarBay.utils.data_to_arrays(
    df_counts[df_counts.rep.=="R1", :]; id_col=:oligo
)[:bc_ids]

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
files = Glob.glob("./output/advi_meanfield_*10000*")

# Initialize dictionary to save outputs
advi_output = Dict()

# Loop through files
for (i, file) in enumerate(files)
    # Load distribution
    setindex!(advi_output, JLD2.load(file), "R$(i)")
    # Convert distribution to tidy dataframe
    df_advi = BarBay.utils.advi_to_df(
        advi_output["R$(i)"]["dist"],
        advi_output["R$(i)"]["var"],
        advi_output["R$(i)"]["ids"];
        genotypes=edits
    )
    # Add result to dictionary
    setindex!(advi_output["R$(i)"], df_advi, "df_advi")
    # Write results into CSV file
    CSV.write(
        "./output/advi_hierarchicalgenotypes_results_R$(i).csv", df_advi
    )
end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate samples from distribution and format into dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples
n_samples = 5_000

# Loop through each replicate
for rep in sort(collect(eachindex(advi_output)))
    println("Replicate $(rep)")
    # Sample from ADVI joint distribution and convert to dataframe
    setindex!(
        advi_output[rep],
        DF.DataFrame(
            Random.rand(
                Distributions.MvNormal(
                    advi_output[rep]["df_advi"].mean,
                    LinearAlgebra.Diagonal(advi_output[rep]["df_advi"].std .^ 2)
                ),
                n_samples
            )',
            advi_output[rep]["df_advi"].varname
        ),
        "df_samples"
    )
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

# Remove previous plot if it exists
rm("./output/figs/advi_logfreqratio_ppc_neutral.pdf"; force=true)

# Loop through repeats
for rep in sort(collect(eachindex(advi_output)))
    println("Generating plot for $(rep)")
    # Compute posterior predictive checks
    ppc_mat = BarBay.stats.logfreq_ratio_popmean_ppc(
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
        df_counts[(df_counts.neutral).&(df_counts.rep.==rep), :];
        freq_col=:freq,
        id_col=:oligo,
        color=:black,
        alpha=0.25,
        linewidth=2
    )

    # Save figure into pdf
    save("./output/figs/temp.pdf", fig)

    # Append pdf
    PDFmerger.append_pdf!(
        "./output/figs/advi_logfreqratio_ppc_neutral.pdf",
        "./output/figs/temp.pdf",
        cleanup=true
    )
end # for

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot example posterior predictive checks
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Remove previous plot if it exists
rm("./output/figs/advi_logfreqratio_ppc_mutants.pdf"; force=true)

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [4, 4]

# List example barcodes to plot
bc_plot = StatsBase.sample(
    unique(df_counts[.!df_counts.neutral, :oligo]), n_row * n_col
)

# Extract unique mutant/fitnes variable name pairs
bc_var = advi_output["R1"]["df_advi"][
    (advi_output["R1"]["df_advi"].vartype.=="bc_fitness"), [:id, :varname]
]

# Generate dictionary from mutant name to fitness value
bc_var_dict = Dict(zip(bc_var.id, bc_var.varname))

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs)))

# Loop through repeats
for rep in sort(collect(eachindex(advi_output)))
    println("Generating plot for $(rep)")
    # Initialize figure
    fig = Figure(resolution=(300 * n_col, 200 * n_row))

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
                df_counts[(df_counts.oligo.==bc_plot[counter]).&(df_counts.rep.==rep), :], :time
            )
            # Extract posterior samples
            df_samples = advi_output[rep]["df_samples"]

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
                time=sort(unique(df_counts[df_counts.rep.==rep, :time]))[2:end]
            )

            # Plot log-frequency ratio of neutrals
            BayesFitUtils.viz.logfreq_ratio_time_series!(
                ax,
                data_bc,
                freq_col=:freq,
                color=:black,
                id_col=:oligo,
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
            counter += 1
        end  # for
    end # for

    # Add title
    Label(fig[1, :, Top()], "Replicate $(rep)", fontsize=20)
    # Add x-axis label
    Label(fig[end, :, Bottom()], "time points", fontsize=20)
    # Add y-axis label
    Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)

    # Save figure to PDF
    save("./output/figs/temp.pdf", fig)

    # Append pdf
    PDFmerger.append_pdf!(
        "./output/figs/advi_logfreqratio_ppc_mutants.pdf",
        "./output/figs/temp.pdf",
        cleanup=true
    )
end # for