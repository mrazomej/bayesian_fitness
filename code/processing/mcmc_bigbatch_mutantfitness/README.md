## `scripts`

- `mcmc_mean_fitness.jl`: Script that runs MCMC on the `#BigBatch` data to infer
  the population mean fitness $\bar{s}_t$ for each pair of contiguous time
  points. Each of the output of this script--one per condition and pair of time
  points--are stored as
  ```
  ./output/$(n_steps)steps_$(n_walkers)walkers/$(batch)_$(hub)hub_$(perturbation)perturbation_$(rep)rep_Tt-Tt+1_meanfitness_mcmcchains.jld
  ```
  This `jld2` file contains three objects:
    - `chain`: The `Turing.jl` chain for the nuisance parameters (standard deviation from
      log-likelihood and barcode frequencies) and the population mean fitness. 
    - `σ_st`: variance for the prior on the population mean fitness.
    - `σ_σt`: variance for the prior on the Log-likelihood variance.