## `scripts`

- `mcmc_mean_fitness.jl`: Script that runs MCMC on the `#BigBatch` data to infer
  the population mean fitness $\bar{s}_t$ for each pair of contiguous time
  points. Each of the output of this script--one per condition and pair of time
  points--are stored as
  ```
  ./output/$(n_steps)steps_$(n_walkers)walkers/$(batch)_$(hub)hub_$(perturbation)perturbation_$(rep)rep_Tt-Tt+1_meanfitness_mcmcchains.jld2
  ```
  This `jld2` file contains three objects:
    - `chain`: The `Turing.jl` chain for the nuisance parameters (standard
      deviation from log-Normal likelihood and neutral lineages barcode
      frequencies) and the population mean fitness. 
    - `σ_st`: standard deviation for the prior on the population mean fitness.
    - `σ_σt`: standard deviation for the prior on the log-Normal likelihood
      standard deviation.

- `mcmc_mutant_fitness_multithread.jl`: Script that runs MCMC on the `#BigBatch`
  data to infer each mutant's fitness $s^{(m)}$. To speed up things, the
  computation is done in a multithreaded fashion, meaning that one mutant is
  inferred on each thread. Because of this, there is only **one chain sampled
  for each of the mutants**. This script requires the output of the
  `mcmc_mean_fitness.jl` script to parametrize the distribution of the mean
  fitness. All chains for a single experimental condition are stored in a single
  `jld2` file
  ```
  ./output/$(n_steps)steps_1walkers/$(batch)_$(hub)hub_$(perturbation)perturbation_$(rep)rep_mutantfitness_mcmcchains.jld2
  ```
  This `jl2` files contain three objects:
    - `chains`: A dictionary where the keys are the `barcodes` of the
      corresponding mutant and the items are the `Turing.jl` chains for the
      nuisance paraeters (standard deviation from log-Normal likelihood and
      mutant barcode frequency) and the mutant relative fitness.
    - `σₛ`: standard deviation for the prior on the mutant fitness.
    - `σₑ`: standard deviation for the prior on the log-Normal likelihood
      standard deviation.