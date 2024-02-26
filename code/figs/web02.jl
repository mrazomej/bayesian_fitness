

# Import package with useful plotting functions for our dataset
import BayesFitUtils

# Import Bayesian inference package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import plotting libraries
using CairoMakie
import ColorSchemes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

# Import library to set random seed
import Random

Random.seed!(42)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_001/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot trajectories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(size=(350, 300))

# Add axis
ax = Axis(fig[1, 1], xlabel="time [dilution cycles]", ylabel="ln(fₜ₊₁/fₜ)")

# Plot mutant barcode trajectories
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    data[.!(data.neutral), :],
    alpha=0.3
)

# Plot neutral barcode trajectories
BayesFitUtils.viz.logfreq_ratio_time_series!(
    ax,
    data[data.neutral, :],
    color=ColorSchemes.Blues_9[end],
)

save("/Users/mrazo/git/BarBay/docs/src/figs/fig02.svg", fig)

fig