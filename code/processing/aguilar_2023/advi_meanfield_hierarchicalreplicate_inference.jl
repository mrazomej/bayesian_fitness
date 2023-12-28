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

# Add oligos by edit
df_edit = DF.combine(
    DF.groupby(df, [:edit, :rep, :time, :neutral]), :count => sum
)

# Rename column
DF.rename!(df_edit, :count_sum => :count)

# Compute sum of each edit
df_sum = DF.combine(DF.groupby(df_edit, [:rep, :time]), :count => sum)

# Add sum to dataframe
DF.leftjoin!(df_edit, df_sum; on=[:rep, :time])

# Compute frequency
df_edit[!, :freq] = df_edit.count ./ df_edit.count_sum

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Obtain priors on expected errors from neutral measurements
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Compute naive priors from neutral strains
naive_priors = BarBay.stats.naive_prior(
    df_edit; rep_col=:rep, pseudocount=1, id_col=:edit
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define ADVI function parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

param = Dict(
    :data => df_edit,
    :outputname => "./output/advi_meanfield_hierarchicalreplicate" *
                   "$(lpad(n_samples, 2, "0"))samples_$(n_steps)steps",
    :model => BarBay.model.replicate_fitness_normal,
    :model_kwargs => Dict(
        :s_pop_prior => s_pop_prior,
        :logσ_pop_prior => logσ_pop_prior,
        :logσ_bc_prior => logσ_bc_prior,
        :s_bc_prior => [0.0, 1.0],
        :logλ_prior => logλ_prior,
    ),
    :rep_col => :rep,
    :id_col => :edit,
    :advi => Turing.ADVI(n_samples, n_steps),
    :opt => Turing.TruncatedADAGrad()
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Perform optimization
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Create output directory
if !isdir("./output/")
    mkdir("./output/")
end # if

# Run inference
println("Running Variational Inference...")
@time BarBay.vi.advi(; param...)