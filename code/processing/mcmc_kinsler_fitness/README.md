# `scripts`

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