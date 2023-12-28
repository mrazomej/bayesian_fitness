##
println("Loading packages...")
# Import project package
import BayesFitUtils

# Import library package
import BarBay

# Import libraries to manipulate data
import DataFrames as DF
import CSV

# Import library to perform Bayesian inference
import Turing

# Import AutoDiff backend
using ReverseDiff
# Impor statistical libraries
import Random
import StatsBase
import Distributions

Random.seed!(42)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI hyerparameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of samples and steps
n_samples = 1
n_steps = 10_000

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Generate output directories
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading the data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

println("Loading data...")

# Import data
df = CSV.read(
    "$(git_root())/data/aguilar_2023/tidy_counts_oligo.csv", DF.DataFrame
)
# Remove repeat "N/A"
df = df[df.rep.!="N/A", :]

# Group data by rep
df_group = DF.groupby(df, :rep)

println("Number of repeats: $(length(df_group))")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Run inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Loop through repeats
Threads.@threads for i in eachindex(df_group)

    # Define filename
    fname = "./output/advi_meanfield_hierarchicalgenotypes_R$(i)rep_" *
            "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps.jld2"

    # Check if file already exists
    if isfile(fname)
        println("Skipping R$(i) since it was already processed")
        continue
    end # if

    # Obtain priors on expected errors from neutral measurements
    println("Computing priors from neutral lineages in R$(i)")

    # Extract data
    data = df_group[i]

    # Compute naive priors from neutral strains
    naive_priors = BarBay.stats.naive_prior(
        data; pseudocount=1, id_col=:oligo
    )

    # Select standard deviation parameters
    s_pop_prior = hcat(
        naive_priors[:s_pop_prior],
        repeat([0.05], length(naive_priors[:s_pop_prior]))
    )

    logσ_pop_prior = hcat(
        naive_priors[:logσ_pop_prior],
        repeat([1.0], length(naive_priors[:logσ_pop_prior]))
    )

    logσ_bc_prior = [StatsBase.mean(naive_priors[:logσ_pop_prior]), 1.0]

    logλ_prior = hcat(
        naive_priors[:logλ_prior],
        repeat([3.0], length(naive_priors[:logλ_prior]))
    )

    # Define ADVI function parameters
    println("Setting relationship between oligos and edits")
    # Define unique oligo-edit pairs
    oligo_edit = unique(data[:, [:oligo, :edit]])

    # Generate dictionary from oligos to edits
    oligo_edit_dict = Dict(oligo_edit[:, :oligo] .=> oligo_edit[:, :edit])

    # Extract list of oligos as they will be used in the inference
    oligo_ids = BarBay.utils.data_to_arrays(data; id_col=:oligo)[:bc_ids]

    # Extract edits in the order they will be used in the inference
    edit_list = [oligo_edit_dict[m] for m in oligo_ids]

    println("Defining parameters for inference")

    param = Dict(
        :data => data,
        :outputname => "./output/advi_meanfield_hierarchicalgenotypes_R$(i)rep_" *
                       "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
        :model => BarBay.model.genotype_fitness_normal,
        :model_kwargs => Dict(
            :s_pop_prior => s_pop_prior,
            :logσ_pop_prior => logσ_pop_prior,
            :logσ_bc_prior => logσ_bc_prior,
            :s_bc_prior => [0.0, 1.0],
            :genotypes => edit_list,
        ),
        :id_col => :oligo,
        :advi => Turing.ADVI(n_samples, n_steps),
        :opt => Turing.TruncatedADAGrad(),
        :fullrank => false
    )

    # Run inference
    println("Running Variational Inference...")
    @time dist = BarBay.vi.advi(; param...)

end # for