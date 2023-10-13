##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import package to revise package
import Revise
# Import project package
import BayesFitUtils

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to list files
import Glob

# Import plotting libraries
using CairoMakie
import ColorSchemes
import PDFmerger

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.pboc_makie!()

import Random

Random.seed!(42)

##

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

# Generate figure dictionary
if !isdir("./output/figs/")
    mkdir("./output/figs/")
end # if

println("Loading data...")

# Load data
df = CSV.read("$(git_root())/data/big_batch/tidy_counts.csv", DF.DataFrame)

##

# Group data
df_group = DF.groupby(df, [:batch, :hub, :perturbation])
# Collect information
df_keys = values.(keys(df_group))

# Remove old version of file
rm("./output/figs/freq_trajectories.pdf", force=true)

# Loop through environments
for (i, data) in enumerate(df_group)
    # Extract metadata
    batch, hub, pert = df_keys[i]
    # Group data by replicate
    data_group = DF.groupby(data, :rep)
    # Define number of plots to generate
    n_plots = ceil(Int64, length(data_group) / 2)
    # Initialize rep counter
    rep_count = 1
    # Loop through plots
    for n = 1:n_plots
        # Initialize figure
        fig = Figure(resolution=(1000, 400))
        # Add axis
        ax1 = Axis(
            fig[1, 1],
            xlabel="time point",
            ylabel="barcode frequency",
            yscale=log10,
            title="$(batch) | $(hub) | $(pert) | R$(rep_count)"
        )
        # Define zero_lim value from data
        zero_lim = minimum(
            data_group[rep_count][data_group[rep_count][:, :freq].>0, :freq]
        ) / 10
        # Plot Mutant barcode trajectories
        BayesFitUtils.viz.bc_time_series!(
            ax1,
            data_group[rep_count][.!data_group[rep_count].neutral, :];
            quant_col=:freq,
            zero_lim=zero_lim,
            zero_label="extinct",
            alpha=0.25,
            linewidth=2
        )

        # Plot Neutral barcode trajectories
        BayesFitUtils.viz.bc_time_series!(
            ax1,
            data_group[rep_count][data_group[rep_count].neutral, :];
            quant_col=:freq,
            zero_lim=zero_lim,
            color=ColorSchemes.Blues_9[end],
            alpha=0.9,
            linewidth=2
        )

        # Update counter
        rep_count += 1

        # Add second axis
        ax2 = Axis(
            fig[1, 2],
            xlabel="time point",
            ylabel="barcode frequency",
            yscale=log10,
            title="$(batch) | $(hub) | $(pert) | R$(rep_count)"
        )
        # Check that counter is within number of reps
        if rep_count > length(data_group)
            # Save figure into pdf
            save("./output/figs/temp.pdf", fig)

            # Append pdf
            PDFmerger.append_pdf!(
                "./output/figs/freq_trajectories.pdf",
                "./output/figs/temp.pdf",
                cleanup=true
            )
            continue
        end # if

        # Define zero_lim value from data
        zero_lim = minimum(
            data_group[rep_count][data_group[rep_count][:, :freq].>0, :freq]
        ) / 10

        # Plot Mutant barcode trajectories
        BayesFitUtils.viz.bc_time_series!(
            ax2,
            data_group[rep_count][.!data_group[rep_count].neutral, :];
            quant_col=:freq,
            zero_lim=zero_lim,
            zero_label="extinct",
            alpha=0.25,
            linewidth=2
        )

        # Plot Neutral barcode trajectories
        BayesFitUtils.viz.bc_time_series!(
            ax2,
            data_group[rep_count][data_group[rep_count].neutral, :];
            quant_col=:freq,
            zero_lim=zero_lim,
            color=ColorSchemes.Blues_9[end],
            alpha=0.9,
            linewidth=2
        )

        # Update counter
        rep_count += 1

        # Save figure into pdf
        save("./output/figs/temp.pdf", fig)

        # Append pdf
        PDFmerger.append_pdf!(
            "./output/figs/freq_trajectories.pdf",
            "./output/figs/temp.pdf",
            cleanup=true
        )
    end #for
end # for
##

# Remove old version of file
rm("./output/figs/logfreqratio_trajectories.pdf", force=true)

# Loop through environments
for (i, data) in enumerate(df_group)
    # Extract metadata
    batch, hub, pert = df_keys[i]

    # Extract and group data by replicate
    data_group = DF.groupby(data, :rep)
    # Define number of plots to generate
    n_plots = ceil(Int64, length(data_group) / 2)
    # Initialize rep counter
    rep_count = 1
    # Loop through plots
    for n = 1:n_plots
        # Initialize figure
        fig = Figure(resolution=(1000, 400))
        # Add axis
        ax1 = Axis(
            fig[1, 1],
            xlabel="time point",
            ylabel="ln(fₜ₊₁/fₜ)",
            title="$(batch) | $(hub) | $(pert) | R$(rep_count)"
        )

        # Plot Mutant barcode trajectories
        BayesFitUtils.viz.logfreq_ratio_time_series!(
            ax1,
            data_group[rep_count][.!data_group[rep_count].neutral, :];
            freq_col=:freq,
            alpha=0.25,
            linewidth=2
        )

        # Plot Neutral barcode trajectories
        BayesFitUtils.viz.logfreq_ratio_time_series!(
            ax1,
            data_group[rep_count][data_group[rep_count].neutral, :];
            freq_col=:freq,
            color=ColorSchemes.Blues_9[end],
            alpha=0.9,
            linewidth=2
        )

        # Update counter
        rep_count += 1

        # Add second axis
        ax2 = Axis(
            fig[1, 2],
            xlabel="time point",
            ylabel="ln(fₜ₊₁/fₜ)",
            title="$(batch) | $(hub) | $(pert) | R$(rep_count)"
        )
        # Check that counter is within number of reps
        if rep_count > length(data_group)
            # Save figure into pdf
            save("./output/figs/temp.pdf", fig)

            # Append pdf
            PDFmerger.append_pdf!(
                "./output/figs/logfreqratio_trajectories.pdf",
                "./output/figs/temp.pdf",
                cleanup=true
            )
            continue
        end # if

        # Plot Mutant barcode trajectories
        BayesFitUtils.viz.logfreq_ratio_time_series!(
            ax2,
            data_group[rep_count][.!data_group[rep_count].neutral, :];
            freq_col=:freq,
            alpha=0.25,
            linewidth=2
        )

        # Plot Neutral barcode trajectories
        BayesFitUtils.viz.logfreq_ratio_time_series!(
            ax2,
            data_group[rep_count][data_group[rep_count].neutral, :];
            freq_col=:freq,
            color=ColorSchemes.Blues_9[end],
            alpha=0.9,
            linewidth=2
        )

        # Update counter
        rep_count += 1

        # Save figure into pdf
        save("./output/figs/temp.pdf", fig)

        # Append pdf
        PDFmerger.append_pdf!(
            "./output/figs/logfreqratio_trajectories.pdf",
            "./output/figs/temp.pdf",
            cleanup=true
        )
    end #for
end # for
