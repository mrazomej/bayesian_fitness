## `ipynb` files

- `bigbatch_data_wrangling.ipynb`: Notebook to turn the raw barcode counts as
  provided by Olivia into a `tidy dataframe`. The output of this notebook is
  stored in
  ```
  bayesian_fitness/data/big_batch/tidy_counts.csv
  ```
- `bayesian_mean_fitness.ipynb`: Notebook used to develop the model that infers
  the mean fitness for each time point using the neutral strains.
- `bayesian_mutant_fitness.ipynb`: Notebook used to develop the model that
  infers the fitness for all adaptive mutants.
- `kinsler_fitness_eda.ipynb`: Exploratory data analysis for the inferences done
  on the manually-curated Kinsler et al., 2020 datasets.