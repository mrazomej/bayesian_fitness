##
println("Loading packages...")

# Load project package
@load_pkg BayesFitUtils

# Import project package
import BayesFitUtils

# Import library package
import BayesFitness

# Import basic math
import StatsBase
import Distributions
import Random

# Import libraries to manipulate data
import DataFrames as DF
import CSV
import MCMCChains

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

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Loading data
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate output directory 
if !isdir("./output/")
    mkdir("./output/")
end # if

# Generate figure dictionary
if !isdir("./output/figs/")
    mkdir("./output/figs/")
end # if

println("Loading data...")

# Import data
data = CSV.read(
    "$(git_root())/data/logistic_growth/data_005/tidy_data.csv", DF.DataFrame
)

# Define number of experimental repeats
n_rep = length(unique(data.rep))

##
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plotting PPC for population mean fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
files = Glob.glob("./output/chain_popmean_fitness_*")

# Define dictionary with corresponding parameters for variables needed for
# the posterior predictive checks
param = Dict(
    :population_mean_fitness => :s̲ₜ,
    :population_std_fitness => :σ̲ₜ,
)

# Define number of posterior predictive check samples
n_ppc = 500

# Define quantiles to compute
qs = [0.05, 0.68, 0.95]

# Define colors
colors = [
    get(ColorSchemes.Greens_9, LinRange(0.25, 1.0, length(qs))),
    get(ColorSchemes.Purples_9, LinRange(0.25, 1.0, length(qs)))
]

# Initialize figure
fig = Figure(resolution=(350 * n_rep, 300))

# Add axis
ax = [
    Axis(
        fig[1, i],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
    ) for i = 1:length(files)
]

# Loop through files
for (i, file) in enumerate(files)
    # Load chain
    chn = JLD2.load(file)["chain"]

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
        chn, n_ppc; param=param
    )

    # Define time
    t = vec(collect(axes(ppc_mat, 2)) .+ 1)

    # Plot posterior predictive checks
    BayesFitUtils.viz.ppc_time_series!(
        ax[i], qs, ppc_mat; colors=colors[i], time=t
    )

    # Plot log-frequency ratio of neutrals
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax[i],
        data[(data.neutral).&(data.rep.==unique(data.rep)[i]), :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2
    )

    ax[i].title = "neutral lineages PPC | Prior | R$(i)"
end # for

# Save figure into pdf
save("./output/figs/mcmc_logfreqratio_ppc_neutral_prior.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Read chain into memory
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = first(Glob.glob("./output/chain_joint_hierarchical*"))

# Load chain
ids, chn = values(JLD2.load(file))
# Remove the string "mut" from mutant names
mut_num = replace.(ids, "mut" => "")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Format tidy dataframe with proper variable names
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# 1. Locate variables
# Locate hyperparameter variables
θ_var = MCMCChains.namesingroup(chn, :θ̲⁽ᵐ⁾)
# Find columns with fitness parameter deviation 
τ_vars = MCMCChains.namesingroup(chn, :τ̲⁽ᵐ⁾)
θ_tilde_vars = MCMCChains.namesingroup(chn, :θ̲̃⁽ᵐ⁾)
# Find columns with mutant fitness error
σ_vars = MCMCChains.namesingroup(chn, :σ̲⁽ᵐ⁾)
# Extract population mean fitness variable names
pop_mean_vars = MCMCChains.namesingroup(chn, :s̲ₜ)
# Extract population mean fitness error variables
pop_std_vars = MCMCChains.namesingroup(chn, :σ̲ₜ)

# 2. Define names based on barcode name and replicate number
# Define barcode names. This will be in the format `#_Rx`
bc_names = vcat([string.(ids) .* "_R$(i)" for i = 1:n_rep]...)
# Define mean fitness variable names this only includes the `R[x]` part to later
# on attach either s̲ₜ or σ̲ₜ
pop_names = vcat([
    "R$i[" .* string.(1:(length(pop_mean_vars)÷n_rep)) .* "]"
    for i = 1:n_rep
]...)

# 3. Convert chain to tidy dataframe
# Convert chain to tidy dataframe
df_chn = DF.DataFrame(chn)

# 4. Compute individual replicate fitness value. This value is not directly
# track by Turing.jl when sampling because it is a derived quantity, not an
# input to the model. We could use the `Turing.generated_quantities` function to
# compute this, but it is simpler to do it directly from the chains.

# Compute individual strains fitness values
s_mat = hcat(repeat([Matrix(df_chn[:, θ_var])], n_rep)...) .+
        (Matrix(df_chn[:, τ_vars]) .* Matrix(df_chn[:, θ_tilde_vars]))

# 5. Insert individual replicate fitness values to dataframe
# Add fitness values to dataframe
DF.insertcols!(df_chn, (Symbol.("s⁽ᵐ⁾_" .* bc_names) .=> eachcol(s_mat))...)

# 6. Rename corresponding variables
# Rename population mean fitness variables
DF.rename!(
    df_chn,
    [
        θ_var .=> Symbol.("θ⁽ᵐ⁾_" .* string.(ids))
        σ_vars .=> Symbol.("σ⁽ᵐ⁾_" .* bc_names)
        pop_mean_vars .=> Symbol.("s̲ₜ_" .* pop_names)
        pop_std_vars .=> Symbol.("σ̲ₜ_" .* pop_names)
    ]
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for neutral lineages in joint inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(300 * 2, 300))

# Add axis
ax = [
    Axis(
        fig[1, j],
        xlabel="time point",
        ylabel="ln(fₜ₊₁/fₜ)",
    ) for j = 1:n_rep
]

# Define number of posterior predictive check samples
n_ppc = 500

# Define quantiles to compute
qs = [0.68, 0.95, 0.05]

# Define colors
colors = get(ColorSchemes.Blues_9, LinRange(0.25, 1.0, length(qs)))

# Loop through repeats
for rep = 1:n_rep
    # Define dictionary with corresponding parameters for variables needed for
    # the posterior predictive checks
    param = Dict(
        :population_mean_fitness => Symbol("s̲ₜ_R$rep"),
        :population_std_fitness => Symbol("σ̲ₜ_R$rep"),
    )

    # Compute posterior predictive checks
    ppc_mat = BayesFitness.stats.logfreq_ratio_mean_ppc(
        df_chn, n_ppc; param=param
    )

    # Define time
    t = vec(collect(axes(ppc_mat, 2)) .+ 1)

    # Plot posterior predictive checks
    BayesFitUtils.viz.ppc_time_series!(
        ax[rep], qs, ppc_mat; colors=colors, time=t
    )

    # Plot log-frequency ratio of neutrals
    BayesFitUtils.viz.logfreq_ratio_time_series!(
        ax[rep],
        data[(data.neutral).&(data.rep.=="R$rep"), :];
        freq_col=:freq,
        color=:black,
        alpha=1.0,
        linewidth=2
    )

    # Set axis title
    ax[rep].title = "log-frequency ratio PPC | R$rep"
end # for

# Save figure into pdf
save("./output/figs/mcmc_logfreqratio_ppc_neutral_posterior.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compute summary statistics for each barcode
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize dataframe to save individual replicate summary statistics
df_summary = DF.DataFrame()

# Find barcode variables
var_names = names(df_chn)[occursin.("s⁽ᵐ⁾", names(df_chn))]

# Define percentiles to include
per = [2.5, 97.5, 16, 84]

# Loop through barcodes
for bc in var_names
    # Extract hyperparameter variable name
    θ_name = replace(split(bc, "_")[2], "bc" => "")
    # Extract fitness chain
    fitness_s = @view df_chn[:, bc]
    # Extract hyperparameter chain
    fitness_θ = @view df_chn[:, "θ⁽ᵐ⁾_$(θ_name)"]

    ## FILTER highest percentiles ##
    fitness_s = fitness_s[
        (fitness_s.≥StatsBase.percentile(fitness_s, 5)).&(fitness_s.≤StatsBase.percentile(fitness_s, 95))
    ]
    fitness_θ = fitness_θ[
        (fitness_θ.≥StatsBase.percentile(fitness_θ, 5)).&(fitness_θ.≤StatsBase.percentile(fitness_θ, 95))
    ]

    # Compute summary statistics
    fitness_summary = Dict(
        :s_mean => StatsBase.mean(fitness_s),
        :s_median => StatsBase.median(fitness_s),
        :s_std => StatsBase.std(fitness_s),
        :s_var => StatsBase.var(fitness_s),
        :s_skewness => StatsBase.skewness(fitness_s),
        :s_kurtosis => StatsBase.kurtosis(fitness_s),
        :θ_mean => StatsBase.mean(fitness_θ),
        :θ_median => StatsBase.median(fitness_θ),
        :θ_std => StatsBase.std(fitness_θ),
        :θ_var => StatsBase.var(fitness_θ),
        :θ_skewness => StatsBase.skewness(fitness_θ),
        :θ_kurtosis => StatsBase.kurtosis(fitness_θ),)

    # Loop through percentiles
    for p in per
        setindex!(
            fitness_summary,
            StatsBase.percentile(fitness_s, p),
            Symbol("s_$(p)_percentile")
        )
        setindex!(
            fitness_summary,
            StatsBase.percentile(fitness_θ, p),
            Symbol("θ_$(p)_percentile")
        )
    end # for

    # Convert to dataframe
    df_fitness = DF.DataFrame(fitness_summary)
    # Add barcode
    df_fitness[!, :id] .= bc
    # Append to dataframe
    DF.append!(df_summary, df_fitness)
end # for

# Add barcode and replicated number as extra columns
DF.insertcols!(
    df_summary,
    :barcode => [split(x, "_")[2] for x in df_summary.id],
    :rep => [split(x, "_")[3] for x in df_summary.id],
)

# Sort by barcode
DF.sort!(df_summary, :barcode)

# Append fitness and growth rate value
DF.leftjoin!(
    df_summary,
    unique(
        data[.!(data.neutral),
            [:barcode, :rep, :fitness, :growth_rate, :growth_rate_exp]]
    );
    on=[:barcode, :rep]
)

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness for individual replicates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="replicate 1 fitness",
    ylabel="replicate 2 fitness",
    title="fitness comparison",
    aspect=AxisAspect(1)
)

# Plot identity line
lines!(ax, [-0.5, 1.75], [-0.5, 1.75], linestyle=:dash, color="black")

# Group data by repeat
df_group = DF.groupby(df_summary, :rep)

# Plot x-axis error bars
errorbars!(
    ax,
    df_group[1].s_median,
    df_group[2].s_median,
    abs.(df_group[1].s_median .- df_group[1][:, Symbol("s_16.0_percentile")]),
    abs.(df_group[1].s_median .- df_group[1][:, Symbol("s_84.0_percentile")]),
    direction=:x,
    linewidth=1.5,
    color=(:gray, 0.5)
)
# Plot y-axis error bars
errorbars!(
    ax,
    df_group[1].s_median,
    df_group[2].s_median,
    abs.(df_group[2].s_median .- df_group[2][:, Symbol("s_16.0_percentile")]),
    abs.(df_group[2].s_median .- df_group[2][:, Symbol("s_84.0_percentile")]),
    direction=:y,
    linewidth=1.5,
    color=(:gray, 0.5)
)

# Plot fitness values
scatter!(ax, df_group[1].s_median, df_group[2].s_median, markersize=5)

save("./output/figs/mcmc_fitness_comparison_replicates.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness with hyperparameter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350 * 2, 350))

# Add axis
ax = [
    Axis(
        fig[1, i],
        xlabel="hyper parameter fitness",
        ylabel="individual replicate fitness",
        aspect=AxisAspect(1),
    ) for i = 1:2
]

# Group data by repeat
df_group = DF.groupby(df_summary, :rep)

# Loop through groups
for (i, df) in enumerate(df_group)
    # Plot identity line
    lines!(ax[i], [-0.5, 1.75], [-0.5, 1.75], linestyle=:dash, color="black")

    # Plot x-axis error bars
    errorbars!(
        ax[i],
        df.θ_median,
        df.s_median,
        abs.(df.θ_median .- df[:, Symbol("θ_16.0_percentile")]),
        abs.(df.θ_median .- df[:, Symbol("θ_84.0_percentile")]),
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.5)
    )
    # Plot y-axis error bars
    errorbars!(
        ax[i],
        df.θ_median,
        df.s_median,
        abs.(df.s_median .- df[:, Symbol("s_16.0_percentile")]),
        abs.(df.s_median .- df[:, Symbol("s_84.0_percentile")]),
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.5)
    )

    # Plot fitness values
    scatter!(ax[i], df.θ_median, df.s_median, markersize=5)

    # Add plot title
    ax[i].title = "replicate R$(i)"

end # for

save("./output/figs/mcmc_fitness_comparison_hyperparameter.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 400))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="true hyper-fitness value",
    ylabel="inferred hyper-fitness",
)

# Plot identity line
lines!(
    ax,
    repeat(
        [[0, 1.75]], 2
    )...;
    color=:black
)

# Add x-axis error bars
errorbars!(
    ax,
    Float64.(df_summary.fitness),
    df_summary.θ_median,
    abs.(df_summary.θ_median .- df_summary[:, "θ_16.0_percentile"]),
    abs.(df_summary.θ_median .- df_summary[:, "θ_84.0_percentile"]),
    color=(:gray, 0.5),
    direction=:y
)

# Plot comparison
scatter!(ax, df_summary.fitness, df_summary.θ_median, markersize=8)

save("./output/figs/mcmc_fitness_true_hyperparameter.pdf", fig)

fig
##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot ECDF distance from median
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(400, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|median - true value|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(df_summary.θ_median .- df_summary.fitness))

save("./output/figs/mcmc_median_true_ecdf.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot Z-score for true fitness value fitting a Normal distribution to chain
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Locate variables for which to fit a normal distribution
θ_names = names(df_chn)[occursin.(Ref("θ⁽ᵐ⁾_mut"), names(df_chn))]

# Fit Gaussian distribution to chains
dist_fit = [
    Distributions.fit(Distributions.Normal, df_chn[:, x]) for x in θ_names
]

# Extract parameters
dist_param = hcat(
    first.(Distributions.params.(dist_fit)),
    last.(Distributions.params.(dist_fit))
)
# Compute Z-score of true fitness values
fitness_zscore = (
    df_summary[df_summary.rep.=="R1", :fitness] .- dist_param[:, 1]
) ./ dist_param[:, 2]
# Initialize figure
fig = Figure(resolution=(400, 300))
# Add axis
ax = Axis(fig[1, 1], xlabel="|z-score|", ylabel="ECDF")
# Plot ECDF
ecdfplot!(ax, abs.(fitness_zscore))

save("./output/figs/mcmc_zscore_ecdf.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot posterior predictive checks for barcodes
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define number of posterior predictive check samples
n_ppc = 500
# Define quantiles to compute
qs = [0.95, 0.675, 0.05]

# Define number of rows and columns
n_row, n_col = [3, 3]

# List example barcodes to plot
bc_plot = StatsBase.sample(ids, n_row * n_col)

# Define colors
colors = [
    get(ColorSchemes.Blues_9, LinRange(0.5, 1, length(qs))),
    get(ColorSchemes.Purples_9, LinRange(0.5, 1, length(qs)))
]

# Initialize figure
fig = Figure(resolution=(300 * n_col, 300 * n_row))

# Initialize plot counter
counter = 1
# Loop through rows
for row in 1:n_row
    # Loop through columns
    for col in 1:n_col
        # Add GridLayout
        gl = fig[row, col] = GridLayout()
        # Add axis
        ax = [Axis(gl[i, 1:6]) for i = 1:n_rep]

        # Loop through replicates
        for rep = 1:n_rep

            # Extract data
            data_bc = DF.sort(
                data[(data.barcode.==bc_plot[counter]).&(data.rep.=="R$rep"),
                    :],
                :time
            )


            # Define dictionary with corresponding parameters for variables needed
            # for the posterior predictive checks
            local param = Dict(
                :mutant_mean_fitness => Symbol(
                    "s⁽ᵐ⁾_$(bc_plot[counter])_R$rep"
                ),
                :mutant_std_fitness => Symbol(
                    "σ⁽ᵐ⁾_$(bc_plot[counter])_R$rep"
                ),
                :population_mean_fitness => Symbol("s̲ₜ_R$rep"),
            )
            # Compute posterior predictive checks
            local ppc_mat = BayesFitness.stats.logfreq_ratio_mutant_ppc(
                df_chn, n_ppc; param=param
            )
            # Plot posterior predictive checks
            BayesFitUtils.viz.ppc_time_series!(
                ax[rep], qs, ppc_mat; colors=colors[rep]
            )

            # Add plot for median (we use the 5 percentile to have a "thicker"
            # line showing the median)
            BayesFitUtils.viz.ppc_time_series!(
                ax[rep], [0.05], ppc_mat; colors=[colors[rep][end]]
            )

            # Add scatter of data
            scatterlines!(ax[rep], diff(log.(data_bc.freq)), color=:black)

            # Add title
            ax[rep].title = "replicate R$rep"
            ax[rep].titlesize = 12
        end # for

        # Hide axis decorations
        hidedecorations!.(ax, grid=false)
        # Set row and col gaps
        rowgap!(gl, 1)

        # Add barcode as title
        Label(
            gl[0, 3:4],
            text="barcode $(bc_plot[counter])",
            fontsize=12,
            justification=:center,
            lineheight=0.9
        )

        # Update counter
        global counter += 1
    end  # for
end # for

# Add x-axis label
Label(fig[end, :, Bottom()], "time points", fontsize=20)
# Add y-axis label
Label(fig[:, 1, Left()], "ln(fₜ₊₁/fₜ)", rotation=π / 2, fontsize=20)

save("./output/figs/mcmc_logfreqratio_ppc_mutant.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Load variational inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define file
file = first(Glob.glob("./output/advi_meanfield*"))

# Load distribution
advi_results = JLD2.load(file)
ids_advi = advi_results["ids"]
dist_advi = advi_results["dist"]
var_advi = advi_results["var"]

# Extract distribution parameters
dist_params = hcat(Distributions.params(dist_advi)...)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Format tidy dataframe with proper variable names
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# Locate variables
# Locate hyperparameter variables
θ_idx = occursin.("θ̲⁽ᵐ⁾", String.(var_advi))
# Find columns with fitness parameter deviation 
τ_idx = occursin.("τ̲⁽ᵐ⁾", String.(var_advi))
θ_tilde_idx = occursin.("θ̲̃⁽ᵐ⁾", String.(var_advi))
# Find columns with mutant fitness error
σ_idx = occursin.("σ̲⁽ᵐ⁾", String.(var_advi))
# Extract population mean fitness variable names
pop_mean_idx = occursin.("s̲ₜ", String.(var_advi))
# Extract population mean fitness error variables
pop_std_idx = occursin.("σ̲ₜ", String.(var_advi))


# Define names based on barcode name and replicate number
# Define barcode names. This will be in the format `#_Rx`
bc_names = vcat([string.(ids) .* "_R$(i)" for i = 1:n_rep]...)
# Define mean fitness variable names this only includes the `R[x]` part to later
# on attach either s̲ₜ or σ̲ₜ
pop_names = vcat([
    "R$i[" .* string.(1:(length(pop_mean_vars)÷n_rep)) .* "]"
    for i = 1:n_rep
]...)

# Convert parameters to dataframe
df_advi = DF.DataFrame(dist_params, ["advi_mean", "advi_std"])
df_advi[!, :var] = names(chn)[1:size(dist_params, 1)]

# Rename corresponding variables
df_advi[θ_idx, :var] .= Symbol.("θ⁽ᵐ⁾_" .* string.(ids))
df_advi[σ_idx, :var] .= Symbol.("σ⁽ᵐ⁾_" .* bc_names)
df_advi[pop_mean_idx, :var] .= Symbol.("s̲ₜ_" .* pop_names)
df_advi[pop_std_idx, :var] .= Symbol.("σ̲ₜ_" .* pop_names)

# Compute individual replicate fitness value. This value is not directly
# track by Turing.jl when sampling because it is a derived quantity, not an
# input to the model. We could use the `Turing.generated_quantities` function to
# compute this, but it is simpler to do it directly from the distributions by
# sampling.

# Define number of samples 
n_sample = 10_000

# Sample θ̲ variables
θ_mat = hcat(
    [
        Random.rand(Distributions.Normal(x...), n_sample)
        for x in eachrow(df_advi[θ_idx, [:advi_mean, :advi_std]])
    ]...
)

# Sample τ̲ variables
τ_mat = exp.(
    hcat(
        [
            Random.rand(Distributions.Normal(x...), n_sample)
            for x in eachrow(df_advi[τ_idx, [:advi_mean, :advi_std]])
        ]...
    )
)

# Sample θ̲̃ variables
θ_tilde_mat = hcat(
    [
        Random.rand(Distributions.Normal(x...), n_sample)
        for x in eachrow(df_advi[θ_tilde_idx, [:advi_mean, :advi_std]])
    ]...
)


# Compute individual strains fitness values
s_mat = hcat(repeat([θ_mat], n_rep)...) .+ (τ_mat .* θ_tilde_mat)

# Compute mean and standard deviation
s_param = hcat(
    [StatsBase.median.(eachcol(s_mat)), StatsBase.std.(eachcol(s_mat))]...
)

# Insert individual replicate fitness values to dataframe
DF.append!(
    df_advi,
    DF.DataFrame(
        hcat(s_param, Symbol.("s⁽ᵐ⁾_" .* bc_names)),
        [:advi_mean, :advi_std, :var]
    )
)

# Locate repeat information
rep = Vector{String}(undef, size(df_advi, 1))
rep[occursin.("R1", String.(df_advi.var))] .= "R1"
rep[occursin.("R2", String.(df_advi.var))] .= "R2"
rep[.!occursin.("R", String.(df_advi.var))] .= "NA"

# Add repeat information
df_advi[!, :rep] = rep

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare inferred population mean fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Locate variable names in MCMC
mcmc_names = names(df_chn)[occursin.("s̲ₜ", names(df_chn))]
# Locate variables in ADVI
advi_idx = occursin.("s̲ₜ", String.(df_advi.var))

# Compute mcmc mean and std
mcmc_mean = StatsBase.mean.(eachcol(df_chn[:, mcmc_names]))
mcmc_std = StatsBase.std.(eachcol(df_chn[:, mcmc_names]))

# Extract advi mean and std
advi_mean = df_advi[advi_idx, :advi_mean]
advi_std = df_advi[advi_idx, :advi_std]


# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="MCMC inference",
    ylabel="ADVI inference",
    title="population mean fitness",
    aspect=AxisAspect(1),
)

# Add identity line
lines!(ax, repeat([[0.2, 1]], 2)..., linestyle=:dash, color=:black)


# add errorbars
errorbars!(
    mcmc_mean,
    advi_mean,
    mcmc_std,
    color=(:gray, 0.5),
    direction=:x
)
# add errorbars
errorbars!(
    mcmc_mean,
    advi_mean,
    advi_std,
    color=(:gray, 0.5),
    direction=:y
)

# Add points
scatter!(
    mcmc_mean,
    advi_mean,
    markersize=8
)


save("./output/figs/advi_vs_mcmc_popmean.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare inferred mutant relative fitness
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Locate variable names in MCMC
mcmc_names = names(df_chn)[occursin.("s⁽ᵐ⁾", names(df_chn))]
# Locate variables in ADVI
advi_idx = occursin.("s⁽ᵐ⁾", String.(df_advi.var))

# Compute mcmc mean and std
mcmc_mean = StatsBase.mean.(eachcol(df_chn[:, mcmc_names]))
mcmc_std = StatsBase.std.(eachcol(df_chn[:, mcmc_names]))

# Extract advi mean and std
advi_mean = df_advi[advi_idx, :advi_mean]
advi_std = df_advi[advi_idx, :advi_std]


# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="MCMC inference",
    ylabel="ADVI inference",
    title="mutant relative fitness",
    aspect=AxisAspect(1),
)

# Add identity line
lines!(ax, repeat([[-0.5, 1.75]], 2)..., linestyle=:dash, color=:black)

# add errorbars
errorbars!(
    mcmc_mean,
    advi_mean,
    mcmc_std,
    color=(:gray, 0.5),
    direction=:x
)
# add errorbars
errorbars!(
    mcmc_mean,
    advi_mean,
    advi_std,
    color=(:gray, 0.5),
    direction=:y
)

# Add points
scatter!(
    mcmc_mean,
    advi_mean,
    markersize=8
)

save("./output/figs/advi_vs_mcmc_mutant.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare inferred mutant relative fitness hyperparameter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Locate variable names in MCMC
mcmc_names = names(df_chn)[occursin.("θ⁽ᵐ⁾", names(df_chn))]
# Locate variables in ADVI
advi_idx = occursin.("θ⁽ᵐ⁾", String.(df_advi.var))

# Compute mcmc mean and std
mcmc_mean = StatsBase.mean.(eachcol(df_chn[:, mcmc_names]))
mcmc_std = StatsBase.std.(eachcol(df_chn[:, mcmc_names]))

# Extract advi mean and std
advi_mean = df_advi[advi_idx, :advi_mean]
advi_std = df_advi[advi_idx, :advi_std]


# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="MCMC inference",
    ylabel="ADVI inference",
    title="relative fitness hyperparameter",
    aspect=AxisAspect(1),
)

# Add identity line
lines!(ax, repeat([[-0.5, 1.75]], 2)..., linestyle=:dash, color=:black)

# add errorbars
errorbars!(
    mcmc_mean,
    advi_mean,
    mcmc_std,
    color=(:gray, 0.5),
    direction=:x
)
# add errorbars
errorbars!(
    mcmc_mean,
    advi_mean,
    advi_std,
    color=(:gray, 0.5),
    direction=:y
)

# Add points
scatter!(
    mcmc_mean,
    advi_mean,
    markersize=8
)

save("./output/figs/advi_vs_mcmc_hyperparameter.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness for individual replicates
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="replicate 1 fitness",
    ylabel="replicate 2 fitness",
    title="fitness comparison",
    aspect=AxisAspect(1)
)

# Plot identity line
lines!(ax, [-0.5, 1.75], [-0.5, 1.75], linestyle=:dash, color="black")

# Group data by repeat
df_group = DF.groupby(df_advi[occursin.("s⁽ᵐ⁾", String.(df_advi.var)), :], :rep)

# Plot x-axis error bars
errorbars!(
    ax,
    df_group[1].advi_mean,
    df_group[2].advi_mean,
    df_group[1].advi_std,
    direction=:x,
    linewidth=1.5,
    color=(:gray, 0.5)
)
# Plot y-axis error bars
errorbars!(
    ax,
    df_group[1].advi_mean,
    df_group[2].advi_mean,
    df_group[2].advi_std,
    direction=:y,
    linewidth=1.5,
    color=(:gray, 0.5)
)

# Plot fitness values
scatter!(ax, df_group[1].advi_mean, df_group[2].advi_mean, markersize=5)

save("./output/figs/advi_fitness_comparison_replicates.pdf", fig)

fig

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Compare median fitness with hyperparameter
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Initialize figure
fig = Figure(resolution=(350 * 2, 350))

# Add axis
ax = [
    Axis(
        fig[1, i],
        xlabel="hyper parameter fitness",
        ylabel="individual replicate fitness",
        aspect=AxisAspect(1),
    ) for i = 1:2
]

# Loop through repeats
for i in 1:n_rep
    # Plot identity line
    lines!(ax[i], [-0.5, 1.75], [-0.5, 1.75], linestyle=:dash, color="black")

    # extract indexes
    θ_idx = occursin.("θ⁽ᵐ⁾", String.(df_advi.var))
    s_idx = (occursin.("s⁽ᵐ⁾", String.(df_advi.var))) .&
            (occursin.("R$(1)", df_advi.rep))
    # Plot x-axis error bars
    errorbars!(
        ax[i],
        df_advi[θ_idx, :advi_mean],
        df_advi[s_idx, :advi_mean],
        df_advi[θ_idx, :advi_std],
        direction=:x,
        linewidth=1.5,
        color=(:gray, 0.5)
    )
    # Plot y-axis error bars
    errorbars!(
        ax[i],
        df_advi[θ_idx, :advi_mean],
        df_advi[s_idx, :advi_mean],
        df_advi[s_idx, :advi_std],
        direction=:y,
        linewidth=1.5,
        color=(:gray, 0.5)
    )

    # Plot fitness values
    scatter!(
        ax[i],
        df_advi[θ_idx, :advi_mean],
        df_advi[s_idx, :advi_mean],
        markersize=5
    )

    # Add plot title
    ax[i].title = "replicate R$(i)"

end # for

save("./output/figs/advi_fitness_comparison_hyperparameter.pdf", fig)

fig

##

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Plot comparison between deterministic and Bayesian inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Generate dictionary from mutant barcode to true fitness value
fit_dict = Dict(zip(df_summary.barcode, df_summary.fitness))

# Extract index for hyperparameter variables
θ_idx = occursin.("θ⁽ᵐ⁾", String.(df_advi.var))

# Initialize figure
fig = Figure(resolution=(350, 350))

# Add axis
ax = Axis(
    fig[1, 1],
    xlabel="true hyper-fitness value",
    ylabel="ADVI inferred hyper-fitness",
)

# Plot identity line
lines!(
    ax,
    repeat(
        [[0, 1.75]], 2
    )...;
    color=:black
)

# Add x-axis error bars
errorbars!(
    ax,
    [
        fit_dict[x] for x in [split(x, "_")[end] for x in
              String.(df_advi[θ_idx, :var])]
    ],
    df_advi[θ_idx, :advi_mean],
    df_advi[θ_idx, :advi_std],
    color=(:gray, 0.5),
    direction=:y
)

# Plot comparison
scatter!(
    ax,
    [
        fit_dict[x] for x in [split(x, "_")[end] for x in
              String.(df_advi[θ_idx, :var])]
    ],
    df_advi[θ_idx, :advi_mean],
    markersize=8
)

save("./output/figs/advi_fitness_true_hyperparameter.pdf", fig)

fig
