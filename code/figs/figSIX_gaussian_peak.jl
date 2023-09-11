##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

import Revise
# Import project package
import BayesFitUtils

# Import basic math
import StatsBase
import Distributions
import LinearAlgebra

# Import plotting libraries
using CairoMakie
import ColorSchemes
import ColorTypes

# Activate backend
CairoMakie.activate!()

# Set PBoC Plotting style
BayesFitUtils.viz.theme_makie!()

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define Gaussian peak function for schematic
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define function
f(x₁, x₂) = 10 * exp(-(x₁^2 + x₂^2))

# Define complete vector function
f̲(x) = [x[1], x[2], f(x[1], x[2])]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Evaluate funciton over 2D grid
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define numebr of points in range
n_grid = 10

# Set range of values where to evaluate jacobian 
x1_grid = x2_grid = LinRange{Float64}(-8, 8, n_grid)

# Set a grid of values for bottom of plot
floor_grid = [0.0 for x in Iterators.product(x1_grid, x2_grid)]

# Define numebr of points in range
n_range = 100

# Set range of values where to evaluate jacobian 
x1 = x2 = LinRange{Float64}(-8, 8, n_range)

# Evaluate function on grid
f_output = hcat(vec([f̲(collect(x)) for x in Iterators.product(x1, x2)])...)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot surface
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 300))

# Add axis for latent space
ax = Axis3(
    fig[1, 1],
    xlabel="x₁",
    ylabel="x₂",
    zlabel="π(x₁,x₂)",
    xypanelcolor="#E6E6EF",
    xzpanelcolor="#E6E6EF",
    yzpanelcolor="#E6E6EF",
    xgridcolor=:white,
    ygridcolor=:white,
    zgridcolor=:white,
    xticksvisible=false,
    xticklabelsvisible=false,
    yticksvisible=false,
    yticklabelsvisible=false,
    zticksvisible=false,
    zticklabelsvisible=false,
)

# Plot surface
surface!(
    ax,
    eachrow(f_output)...,
    shading=false,
    colormap=Reverse(:magma),
    alpha=1.0
)
# Plot discrete grid on bottom
wireframe!(
    ax,
    x1_grid,
    x2_grid,
    floor_grid,
    color=:black
)

save("$(git_root())/doc/figs/figSIX_gaussian_peak.pdf", fig)
save("$(git_root())/doc/figs/figSIX_gaussian_peak.png", fig)

fig