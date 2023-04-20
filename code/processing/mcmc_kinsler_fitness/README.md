# `scripts`

- `viz_trajectories.jl`: Script used to generate the frequency trajectories as
  well as the log-frequency ratio trajectories for all the environments in the
  Kinsler et al., 2020 dataset. The output of this script are PDF files saved in
  the `./output/figs/` directory.
- `mcmc_mean_fitness.jl`: Script using the `BayesFitness.jl` inference pipeline
  to sample out of the population mean fitness posterior distribution for all
  the environments in the Kinsler et al., 2020 dataset. The output of this
  script is stored in `./output/`, with subdirectories of the form
  ```
  ./output/$(env)_R$(rep)/
  ```
  The files on each of the subdirectories are stored as
  ```
  ./output/$(env)_R$(rep)/$(env)_R$(rep)_meanfitness_$(t)-$(t+1).jld
  ```
- `mcmc_mutant_fitness.jl`: Script using the `BayesFitness.jl` inference
  pipeline to sample out of the mutants relative fitness posterior distribution
  for all environments in the Kinsler et al., 2020 dataset. The output of this
  script is stored in `./output/`, with subdirectories of the form
  ```
  ./output/$(env)_R$(rep)/
  ```
  The files on each of the subdirectories are stored as
  ```
  ./output/$(env)_R$(rep)/$(env)_R$(rep)_mutantfitness_$(mutant_id).jld
  ```
- `mcmc_mutant_fitness.jl`: Script using the `BayesFitness.jl` inference
  pipeline to sample out of the mutants relative fitness posterior distribution
  for all environments in the Kinsler et al., 2020 dataset ignoring all
  measurements defined as `t=0`. The output of this script is stored in
  `./output/`, with subdirectories of the form
  ```
  ./output/$(env)_R$(rep)/
  ```
  The files on each of the subdirectories are stored as
  ```
  ./output/$(env)_R$(rep)/$(env)_R$(rep)_mutantfitness_rmT0_$(mutant_id).jld
  ```