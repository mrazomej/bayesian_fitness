println("Loading packages...")



# Import project package
import BayesFitUtils

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to list files
import Glob

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of generations per cycle
n_gen = 4

# Define number of neutral lineages
n_neutral = 50
# Define total number of lineages
n_lin = 1_000
# Define number of mutants
n_mut = n_lin - n_neutral

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading the data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_015/tidy_data.csv", DF.DataFrame
)

# Convert data to arrays for FitSeq2 -i input
bc_seq_mat = BarBay.utils.data_to_arrays(data)[:bc_count]

# Read table with number of cells
bc_cell = Matrix(
    CSV.read(
        "./output/fitseq_sim_output_EvoSimulation_Bottleneck_Cell_Number.csv",
        DF.DataFrame,
        header=false
    )
)

# Initialize empty dataframe for CSV FitSeq2 -t input
bc_cell_mat = DF.DataFrame()
# Sum number of cells pert time point
bc_cell_mat[!, :count] = vec(sum(bc_cell, dims=1))

# Transform first column to the number of generations in experiment
bc_cell_mat[!, :time] = (collect(1:size(bc_cell_mat, 1)) .* n_gen) .- n_gen

# Define paths for outputs
seq_path = "./output/tmp_bc_seq.csv"
cell_path = "./output/tmp_bc_cell.csv"

# Save matrices as CSV
CSV.write(seq_path, DF.DataFrame(bc_seq_mat', :auto), writeheader=false)
CSV.write(cell_path, bc_cell_mat[:, [:time, :count]], writeheader=false)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Run FitSeq2 using `run` command
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Running FitSeq2.0 on simulated data...")

# Define the path to the Python executable in Conda environment
python_executable = "/Users/mrazo/miniconda3/envs/fitseq2/bin/python"
# Define the path to Python script
script_path = "/Users/mrazo/git/Fit-Seq2.0/main_code/fitseq2_run.py"
# Define output prefix
output_prefix = "./output/fitseq2"

# Construct the command
cmd = `$(python_executable) 
    $(script_path) 
    -i $(seq_path) 
    -t $(cell_path)
    -o $(output_prefix)`

# Run the command
run(cmd)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Convert FitSeq2 output to tidy dataframe
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Converting output to tidy dataframe")

# Define barcode name
bc_names = [
    ["neutral$(lpad(i, 3,  "0"))" for i = 1:n_neutral]
    ["mut$(lpad(i, 3,  "0"))" for i = 1:n_mut]
]

# Read results
df_fitseq = CSV.read("$(output_prefix)_FitSeq_Result.csv", DF.DataFrame)

# Rename to lowercase
DF.rename!(df_fitseq, lowercase.(names(df_fitseq)))

# Append barcode name
df_fitseq[!, :id] = bc_names

# Remove unnecessary files
files = vcat(
    [Glob.glob("./output/tmp_*"), Glob.glob("$(output_prefix)*")]...
)

# Loop through files to be removed
for f in files
    # Remove files
    rm(f)
end # for

# Save output as a CSV file
CSV.write("./output/fitseq2_inference.csv", df_fitseq)
