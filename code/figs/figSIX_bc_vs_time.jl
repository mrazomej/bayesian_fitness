##
println("Loading packages...")

import Revise
# Import project package
import BayesFitUtils

# Import basic math
import StatsBase
import Distributions
import LinearAlgebra

# Import packages to read data
import CSV
import DataFrames as DF

# Import plotting libraries
using CairoMakie
import ColorSchemes
import ColorTypes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define path to data
data_dir = "$(git_root())/code/processing/" *
           "data012_logistic_100-100000bc_01env_01rep/output"

# Load time table
df_time = CSV.read("$(data_dir)/times.csv", DF.DataFrame)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot number of barcodes vs time
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(size=(350, 300))

# Add axis
ax = Axis(fig[1, 1], xlabel="time [hours]", ylabel="# unique barcodes")

# Plot data
scatterlines!(ax, df_time.n_bc, df_time.time ./ 3600)

save("$(git_root())/doc/figs/figSIX_bc_vs_time.pdf", fig)
save("$(git_root())/doc/figs/figSIX_bc_vs_time.png", fig)

fig