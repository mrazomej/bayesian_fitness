# `scripts`

- `sim_hierarchical_data002.jl`: Script to generate simulated datasets for
  experimental repeats to be analyzed with a hierarchical model.
- `mcmc_freq.jl`: Script to sample the barcode frequency posterior distribution
  to be used as priors for the full joint inference.
- `mcmc_popmean_fitness.jl`: Script to sample the population mean fitness
  posterior distribution to be used as priors for the full joint inference.
- `mcmc_joint_hierarchical_fitness.jl`: Script to sample the full joint
  posterior distribution for all parameters assuming a hierarchical model that
  connects the experimental replicates.
- `mcmc_joint_fitness.jl`: Script to sample the full posterior distribution for
  all parameters taking replicates as independent experiments.