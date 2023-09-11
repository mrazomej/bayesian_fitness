# `scripts`

- `sim_data007.jl`: Script to simulate dataset with a hierarchical structure at
  the genotype level. This is, there are several barcodes that belong to a
  single genotype, thus, have the same fitness value. The outputs of this script
  are stored in the `data/logistic_growth/data_007/` directory.
- `advi_meanfield_hierarchicalgenotype_joint_inference.jl`: Script to perform
  variational inference using a hierarchical model at the level of the
  genotypes.
- `viz_inference.jl`: Series of plotting functions to visualize the performance
  of the inference pipeline.
- `advi_meanfield_joint_inference.jl`: Script to perform variational inference
  using a one-dataset model for each barcode
- `advi_meanfield_pool_inference.jl`: Script to perform variational inference
  using a one-dataset model with all barcodes pooled by genotype.