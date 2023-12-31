#
println("Loading packages...")



import Revise
# Import project package
import BayesFitUtils
# Import package for Bayesian inference
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import Glob

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

# Import library to set random seed
import Random

Random.seed!(42)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading simulated dataset...")
# Import data
df_counts = CSV.read(
    "$(git_root())/data/logistic_growth/data_002/tidy_data.csv", DF.DataFrame
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load ADVI results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading ADVI results...")

# Define file
file = "$(git_root())/code/processing/data002_logistic_1000bc_01env_02rep/" *
       "output/advi_meanfield_01samples_5000steps.csv"

# Convert results to tidy dataframe
df_advi = CSV.read(file, DF.DataFrame)

# Split variables by replicate
rep_vars = Dict(
    Symbol(rep) => df_advi[df_advi.rep.==rep, :varname]
    for rep in unique(df_advi.rep)
)
# Remove "N/A" from dictionary
delete!(rep_vars, Symbol("N/A"))

# Extract bc fitness values
df_fitness = df_advi[(df_advi.vartype.=="bc_fitness"), :]

# Extract and append hyperfitness values for each fitness value
DF.leftjoin!(
    df_fitness,
    DF.rename(
        df_advi[(df_advi.vartype.=="bc_hyperfitness"),
            [:mean, :std, :varname, :id]],
        :mean => :mean_h,
        :std => :std_h,
        :varname => :varname_h
    );
    on=:id
)

# Extract and append ground truth fitness and hyperfitness values
DF.leftjoin!(
    df_fitness,
    DF.rename(
        unique(
            df_counts[.!(df_counts.neutral),
                [:barcode, :fitness, :hyperfitness, :rep]]
        ),
        :barcode => :id
    );
    on=[:id, :rep]
)

# Load single-experiment replicates
# -----------------------------------------------------------------------------

println("Loading ADVI results for individual replicates...")

# Define file
files = sort(
    Glob.glob(
        "$(git_root())/code/processing/"[2:end] *
        "data002_logistic_1000bc_01env_02rep/output/advi_meanfield*rep*5000*",
        "/"
    )
)

# Initialize empty dataframe
df_advi_single = DF.DataFrame()

# Loop through files
for file in files
    # Extract replicate information
    rep = replace(split(split(file, "/")[end], "_")[3], "rep" => "")

    # read dataframe
    df_tmp = CSV.read(file, DF.DataFrame)
    # Replace replicate column
    df_tmp[:, :rep] .= rep

    # Append to DataFrame
    DF.append!(df_advi_single, df_tmp)
end # for

# Extract and append hyperfitness values for each fitness value
DF.leftjoin!(
    df_fitness,
    DF.rename(
        df_advi_single[(df_advi_single.vartype.=="bc_fitness"),
            [:mean, :std, :varname, :id, :rep]],
        :mean => :mean_single,
        :std => :std_single,
        :varname => :varname_single
    );
    on=[:id, :rep]
)

# Add column with mean between replicates for singled dataset model
DF.leftjoin!(
    df_fitness,
    DF.combine(
        DF.groupby(df_fitness, :id), :mean_single => StatsBase.mean
    );
    on=:id
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate figure
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(size=(350 * 2, 350 * 2))

# Add grid layout for datasets
gl = fig[1, 1] = GridLayout()

# Add axis
ax = [
    Axis(
        gl[i, j],
        aspect=AxisAspect(1)
    ) for i = 1:2 for j = 1:2
]

# -----------------------------------------------------------------------------

# Add identity line
lines!(
    ax[1],
    repeat([[1.75, -0.5,]], 2)...,
    linestyle=:dash,
    color="black"
)

# Plot erorr bars for fitness comparison between replicates in hierarchical
# model
errorbars!(
    ax[1],
    df_fitness[(df_fitness.rep.=="R1"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :mean],
    df_fitness[(df_fitness.rep.=="R1"), :std],
    direction=:x,
    color=(:gray, 0.5)
)
errorbars!(
    ax[1],
    df_fitness[(df_fitness.rep.=="R1"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :std],
    direction=:y,
    color=(:gray, 0.25)
)

# Plot mean for fitness comparison between replicates in hierarchical model
scatter!(
    ax[1],
    df_fitness[(df_fitness.rep.=="R1"), :mean],
    df_fitness[(df_fitness.rep.=="R2"), :mean],
    markersize=5
)

ax[1].title = "hierarchical model"
ax[1].xlabel = "replicate R1 fitness"
ax[1].ylabel = "replicate R2 fitness"

# -----------------------------------------------------------------------------

# Add identity line
lines!(
    ax[2],
    repeat([[1.75, -0.5,]], 2)...,
    linestyle=:dash,
    color="black"
)

# Plot erorr bars for fitness comparison between replicates in hierarchical
# model
errorbars!(
    ax[2],
    df_fitness[(df_fitness.rep.=="R1"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :mean_single],
    df_fitness[(df_fitness.rep.=="R1"), :std_single],
    direction=:x,
    color=(:gray, 0.5)
)
errorbars!(
    ax[2],
    df_fitness[(df_fitness.rep.=="R1"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :std_single],
    direction=:y,
    color=(:gray, 0.25)
)

# Plot mean for fitness comparison between replicates in hierarchical model
scatter!(
    ax[2],
    df_fitness[(df_fitness.rep.=="R1"), :mean_single],
    df_fitness[(df_fitness.rep.=="R2"), :mean_single],
    markersize=5,
    color=ColorSchemes.seaborn_colorblind[2]
)

ax[2].title = "single dataset model"
ax[2].xlabel = "replicate R1 fitness"
ax[2].ylabel = "replicate R2 fitness"

# -----------------------------------------------------------------------------

# Add identity line
lines!(
    ax[3],
    repeat(
        [[
            minimum(df_fitness.mean) * 1.1,
            maximum(df_fitness.mean) * 1.1,
        ]],
        2
    )...,
    linestyle=:dash,
    color="black"
)

# Extract data
data = df_fitness[df_fitness.rep.=="R1", :]

# Plot hierarchical hyperfitness vs ground truth
errorbars!(
    ax[3],
    data.hyperfitness,
    data.mean_h,
    data.std_h,
    color=(:gray, 0.25)
)
scatter!(
    ax[3],
    data.hyperfitness,
    data.mean_h,
    label="hierarchical\nhyper-fitness",
    markersize=5,
)

# Plot single-dataset model vs ground turth
scatter!(
    ax[3],
    data.hyperfitness,
    data.mean_single_mean,
    label="⟨single fitness⟩",
    markersize=5
)

# Add legend
axislegend(ax[3], position=:rb, labelsize=13, framevisible=false)

# Label axis
ax[3].xlabel = "ground truth hyper-fitness"
ax[3].ylabel = "inferred parameter"

# -----------------------------------------------------------------------------

# Extract data
data = df_fitness[df_fitness.rep.=="R1", :]

# Plot ECDF for hierarchical model
ecdfplot!(
    ax[4],
    abs.(data.mean_h .- data.hyperfitness),
    label="hierarchical model"
)

# Plot ECDF for single-dataset model
ecdfplot!(
    ax[4],
    abs.(data.mean_single_mean .- data.hyperfitness),
    label="single dataset model"
)

# Add legend
axislegend(ax[4], position=:rb, labelsize=13, framevisible=false)

# Label axis
ax[4].xlabel = "|mean - ground truth hyper-fitness|"
ax[4].ylabel = "ECDF"

# Add label
Label(
    gl[1, 1, TopLeft()], "(A)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add label
Label(
    gl[1, 2, TopLeft()], "(B)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add label
Label(
    gl[2, 1, TopLeft()], "(C)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Add label
Label(
    gl[2, 2, TopLeft()], "(D)",
    fontsize=26,
    padding=(0, 5, 5, 0),
    halign=:right
)

# Set spacing
rowgap!(gl, -10)
colgap!(gl, -20)

save("$(git_root())/doc/figs/fig05.pdf", fig)
save("$(git_root())/doc/figs/fig05.png", fig)

fig