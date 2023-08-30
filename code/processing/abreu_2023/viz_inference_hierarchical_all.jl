##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import project package
import BayesFitUtils

# Import library package
import BayesFitness

# Import basic math
import LinearAlgebra
import StatsBase
import Distributions
import Random

# Import iterator tools
import Combinatorics

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

println("Loading data..\n")

# Import data
df_counts = CSV.read(
    "$(git_root())/data/abreu_2023/tidy_data.csv", DF.DataFrame
)

# Define unique environments
env_unique = unique(df_counts.env)

# Define color for environments to keep consistency
env_colors = Dict(env_unique .=> ColorSchemes.tableau_10[1:3])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Process individual datasets
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Select columns that indicate the dataset
df_idx = df_counts[:, 12:end]

# Enumerate the columns (needed to run things in parallel smoothly)
n_col = size(df_idx, 2)

# for

# Extract data
data = df_counts[values(df_idx[:, col]), 1:11]

# Compile directory name
out_dir = "./output/$(first(data.condition))_$(names(df_idx)[col])"

# Generate figure dictionary if it doesn't exist
fig_dir = "$(out_dir)/figs"
if !isdir(fig_dir)
    mkdir(fig_dir)
end # if

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define list of environments
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Group data by replicate
data_rep_group = DF.groupby(data, :rep)
# Define number of time points per replicate
n_rep_time = [length(unique(d[:, :time])) for d in data_rep_group]

if length(unique(n_rep_time)) == 1
    # Define environment cycles
    envs = collect(unique(data[:, [:time, :env]])[:, :env])
else
    # Obtain list of environments per replicate
    envs = [
        collect(unique(d[:, [:time, :env]])[:, :env])
        for d in data_rep_group
    ]
end # if

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read ADVI results
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

advi_results = JLD2.load(first(Glob.glob("$(out_dir)/*3000*jld2")))

# Convert results to tidy dataframe
df_advi = BayesFitness.utils.advi_to_df(
    advi_results[:dist],
    advi_results[:var],
    advi_results[:ids];
    n_rep=length(unique(data.rep)))

# end # for