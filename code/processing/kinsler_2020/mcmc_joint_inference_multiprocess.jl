#

# Activate environment
@load_pkg BayesFitUtils

# Define number of processes
n_procs = 4

# Import Distributed to add processes and the @everywhere macro.
import Distributed

println("Adding multiple processess...\n")
# Add processes
Distributed.addprocs(n_procs; exeflags="--project=$(Base.active_project())")

println("Number of processes: $(Distributed.nprocs())")
println("Number of workers: $(Distributed.nworkers())\n")

println("Importing packages on all processess...\n")
# Import packages everywhere
Distributed.@everywhere begin
    # Import library package
    import BarBay

    # Import MCMC-related packages
    import Turing
    using ReverseDiff

    # Set AutoDiff backend to ReverseDiff.jl for faster computation
    Turing.setadbackend(:reversediff)
    # Allow system to generate cache to speed up computation
    Turing.setrdcache(true)

    # Import libraries to manipulate data
    import DataFrames as DF
    import CSV

    # Import library to save and load native julia objects
    import JLD2

    # Import library to list files
    import Glob

    import Random

    Random.seed!(42)

    # Define inference parameters
    n_walkers = 4
    n_steps = 1000
end # @everywhere

##

# Import data
df = CSV.read(
    "$(git_root())/data/kinsler_2020/tidy_counts_no_anc.csv", DF.DataFrame
)

# Read datasets visual evaluation info
df_include = CSV.read(
    "$(git_root())/data/kinsler_2020/exp_include.csv", DF.DataFrame
)

# Upload dataset to all processes
@eval Distributed.@everywhere df = $df
@eval Distributed.@everywhere df_include = $df_include

##

# Loop through datasets
for i = 1:size(df_include, 1)
    # Extract info
    env, rep, rm_T0 = collect(df_include[i, :])
    # Extract data
    data = df[(df.env.==env).&(df.rep.==rep), :]

    println("processing $(env) | $(rep)")

    # Define function parameters
    param = Dict(
        :data => data,
        :n_walkers => n_walkers,
        :n_steps => n_steps,
        :outputname => "./output/kinsler_$(env)env_$(rep)rep_$(rm_T0)rmT0",
        :model => BarBay.model.fitness_lognormal,
        :sampler => Turing.NUTS(0.65),
        :ensemble => Turing.MCMCDistributed(),
        :rm_T0 => rm_T0,
        :verbose => true,
    )

    # Run inference
    println("Running Inference for group $(i)...")

    try
        @time BarBay.mcmc.mcmc_joint_fitness(; param...)
    catch
        @warn "Group $(i) was already processed"
    end

end # for
