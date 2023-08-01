# `scripts`

- `viz_trajectories.jl`: Script used to generate the frequency trajectories as
  well as the log-frequency ratio trajectories for all the environments in the
  Kinsler et al., 2020 dataset. The output of this script are PDF files saved in
  the `./output/figs/` directory.
- `advi_meanfield_joint_inference.jl`: Script used to fit a variational
  posterior distribution to all Kinsler et al. datasets using the neutral
  lineages to determine the prior on the nuisance parameters for the likelihood.
  The output of this script is stored in
  `./output/advi_meanfield_joint_inference`, with filenames of the form
  ```
  kinsler_$(env)env_$(rep)rep_$(rmT0)rmT0_$(n_samples)samples_$(n_steps)steps.jld2
  ```
- `mcmc_popmean_fitness.jl`: Script using the `BayesFitness.jl` inference
  pipeline to sample out of the joint population mean fitness posterior
  distribution for all the environments in the Kinsler et al., 2020 dataset
  using only the neutral lineages. The output of this script is stored in
  `./output/popmean_fitness`, with filenames of the form
  ```
  kinsler_$(env)env_$(rep)rep_$(rmT0)rmT0_$(n_steps)steps_$(n_walkers)walkers.jld2
  ```
- `mcmc_freq.jl`: Script using the `BayesFitness.jl` inference pipeline to
  sample out of the joint barcode frequency posterior distribution for all the
  environments in the Kinsler et al., 2020 dataset using only the neutral
  lineages. The output of this script is stored in `./output/bc_freq`,
  with filenames of the form
  ```
  kinsler_$(env)env_$(rep)rep_$(rmT0)rmT0_$(n_steps)steps_$(n_walkers)walkers.jld2
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
- `mcmc_joint_inference_multiprocess.jl`: Script using the `BayesFitness.jl`
  inference pipeline to infer the **full** joint distribution for all neutral
  and mutant lineages at all time points for a set of manually-selected
  environments in the Kinsler et al., 2020 dataset. The inference is performed
  using the `Distributed.jl` package for multiprocessing, meaning that the
  sampling ensemble for `Turing.jl` is `Turing.Distributed()`. The output of
  this script is stored in `./output/` as
  ```
  ./output/kinsler_$(env)env_$(rep)rep_$(rmT0)rmT0.jld2
  ```
  where the last entry indicates if `T0` was removed from the dataset when
  performing the inference.
- `mcmc_joint_inference_multithread.jl`: Script using the `BayesFitness.jl`
  inference pipeline to infer the **full** joint distribution for all neutral
  and mutant lineages at all time points for a set of manually-selected
  environments in the Kinsler et al., 2020 dataset. The inference is performed
  using the `Threads.jl` package for multithreading, meaning that the sampling
  ensemble for `Turing.jl` is `Turing.Threads()`. The output of this script is
  stored in `./output/` as
  ```
  ./output/kinsler_$(env)env_$(rep)rep_$(rmT0)rmT0.jld2
  ```
  where the last entry indicates if `T0` was removed from the dataset when
  performing the inference.
- `mcmc_joint_inference_multiprocess_multithread.jl`: Script using the
  `BayesFitness.jl` inference pipeline to infer the **full** joint distribution
  for all neutral and mutant lineages at all time points for a set of
  manually-selected environments in the Kinsler et al., 2020 dataset. The
  inference is performed in a hybrid format where multiple datasets are
  processed using the `Distributed.@distribured` macro on a for-loop, with each
  of them sampled in a multithread fashion using `Turing.Threads()` as the
  sampling ensemble. The output of this script is stored in `./output/` as
  ```
  ./output/kinsler_$(env)env_$(rep)rep_$(rmT0)rmT0.jld2
  ```
  where the last entry indicates if `T0` was removed from the dataset when
  performing the inference.
- `viz_joint_inference.jl`: Script to generate visualizations for the inferences
  from the "`_joint_`" scripts. This script plots the posterior predictive
  checks for the neutral lineages as well as a selected group of mutant
  lineages.