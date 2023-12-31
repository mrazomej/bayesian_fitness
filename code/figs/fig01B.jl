##
println("Loading packages...")



import Revise
# Import project package
import BayesFitUtils

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_001/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot barcode trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Plotting barcode trajectories")

# Generate color gradient to use for mutants according to fitness value
color_grad = cgrad(ColorSchemes.Blues, unique(data.fitness))

# Collect colors
color = [x for x in color_grad]

# Initialize figure
fig = Figure(size=(400, 250))

# Add axis
ax = Axis(fig[1, 1], xlabel="time [dilution cycles]", ylabel="barcode frequency", yscale=log10)

# %%% Barcode trajectories %%% #

# Import personal color palette
color_palette = BayesFitUtils.viz.colors()

# Plot mutant barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax,
    data[.!(data.neutral), :],
    zero_lim=0,
    quant_col=:freq,
    alpha=0.5,
    color=color,
)

# Plot neutral barcode trajectories
BayesFitUtils.viz.bc_time_series!(
    ax,
    data[data.neutral, :],
    zero_lim=0,
    quant_col=:freq,
    color=color_palette[:pale_black]
)

# Set axis limits
xlims!(ax, [1, 5])

# Add colorbar
Colorbar(
    fig[1, 2],
    colormap=color_grad,
    limits=(minimum(data.fitness), maximum(data.fitness)),
    label="relative fitness"
)

save("$(git_root())/doc/figs/fig01B.pdf", fig)
