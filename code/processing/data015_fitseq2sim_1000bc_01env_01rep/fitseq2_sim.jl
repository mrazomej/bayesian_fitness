println("Loading packages...")

# Import project package
import BayesFitUtils

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to list files
import Glob

# Import basic statistical functions
import StatsBase
import Distributions
import Random


# Set random seed
Random.seed!(42)
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define simulation metadata
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of neutral lineages
n_neutral = 50
# Define total number of lineages
n_lin = 1_000
# Define number of mutants
n_mut = n_lin - n_neutral

# Define number of generations per cycle
n_gen = 4

# Define number of cycles
n_growth_cycle = 5

# Define average number of reads per barcode
reads = 1_000

# Define parameters for initial number of cells according to
# n₀ ~ Gamma(α, β)
α = 20
β = 0.2

# Define parameters for fitness distribution according to
# λ ~ SkewNormal(µ, σ, skew)
µ = 0.0
σ = 0.225
skew = 3

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate output directory
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate output directory if it doesn't exist
if !isdir("./output/")
    mkdir("./output/")
end # if

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate `--t_seq` csv file
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define output path
t_seq_path = "./output/t_seq.csv"

# Define time points
time = collect(1:n_growth_cycle) .* n_gen .- n_gen

# Generate DataFrame
df_tseq = DF.DataFrame(hcat([time, repeat([reads], length(time))]...), :auto)

CSV.write(t_seq_path, df_tseq, writeheader=false)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute initial number of cells and Malthusian fitness
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define output path
fitness_path = "./output/fitness.csv"

# Sample initial number of cells
n̲₀ = Random.rand(Distributions.Gamma(α, 1 / β), n_lin)

# Sample fitness values
λ̲ = [
    repeat([0], n_neutral);
    sort(Random.rand(Distributions.SkewNormal(µ, σ, skew), n_mut))
]

df_fitness = DF.DataFrame(hcat([λ̲, n̲₀]...), :auto)

CSV.write(fitness_path, df_fitness, writeheader=false)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Run simulation using FitSeq2.0
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


println("Running FitSeq2.0 on simulated data...")

# Define the path to the Python executable in Conda environment
python_executable = "/Users/mrazo/miniconda3/envs/fitseq2/bin/python"
# Define the path to Python script
script_path = "/Users/mrazo/git/Fit-Seq2.0/main_code/fitseqsimu_run.py"
# Define output prefix
output_file = "./output/fitseq_sim_output"

# Construct the command
cmd = `$(python_executable) 
    $(script_path) 
    --t_seq $(t_seq_path)
    --fitness $(fitness_path)
    -o $(output_file)`

# Run the command
run(cmd)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Turn output into tidy dataframe
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Search for file with reads per barcode
read_files = first(Glob.glob("$(output_file)*EvoSimulation_Read_Number.csv"))

# Read files
df_reads = CSV.read(read_files, DF.DataFrame, header=false)

# Define barcode name
bc_names = [
    ["neutral$(lpad(i, 3,  "0"))" for i = 1:n_neutral]
    ["mut$(lpad(i, 3,  "0"))" for i = 1:n_mut]
]

# Generate dataframe with columns as the barcode names
data = DF.DataFrame(Matrix(df_reads)', bc_names)

# Add time column
data[!, :time] = 1:size(data, 1)

# Convert to tidy dataframe
data = DF.stack(data, bc_names)

# Rename columns
DF.rename!(data, :variable => :barcode, :value => :count)

# Add neutral index column
data[!, :neutral] = occursin.("neutral", data.barcode)

# Build dataframe with count sum
data_sum = DF.combine(DF.groupby(data, :time), :count => sum)
DF.leftjoin!(data, data_sum; on=:time)

# Add frequency colymn
data[!, :freq] = data.count ./ data.count_sum

# Generate dataframe with "true" fitness
df_fit = DF.DataFrame(hcat([bc_names, λ̲]...), [:barcode, :fitness])

# Add fitness and growth rate information
DF.leftjoin!(data, df_fit; on=:barcode)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 
# Save data to memory
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% # 

# Defne output directory
out_dir = "$(git_root())/data/logistic_growth/data_015"

# Check if output directory exists
if !isdir(out_dir)
    mkdir(out_dir)
end # if

# Define text to go into README
readme = """
# `$(@__FILE__)`
## Number of mutant barcodes
`n_mut = $(n_mut)`
## number of neutral barcodes
`n_neutral = $(n_neutral)`
# Number of generations
`n_gen = $(n_gen)`
"""

# Write README file into memory
open("$(out_dir)/README.md", "w") do file
    write(file, readme)
end

CSV.write("$(out_dir)/tidy_data.csv", data)