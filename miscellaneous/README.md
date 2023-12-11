# `startup.jl`

This directory contains a `Julia` script `startup.jl` that contains two
functions used throughout this repository:

- `git_root()`: Function to find the root environment of a Git repository from
  within any subdirectory in the repository.

- `load_pkg(pkg)`: Function to find the home directory of a `GitHub` repository
to load the package for that particular project.

The functions are simple shortcuts to easily navigate throughout the repository,
activating the corresponding environment when running the code. To add these
functions on your computer, follow these steps:

1. Copy the `startup.jl` file in this repository to the following directory (for
   `Unix`-based OS):
```
 ~/.julia/config/
```
This will load these functions when `Julia` is launched. If you created a
`startup.jl` file before, you can simply copy and paste these functions to such
existing file.

2. Make sure `Suppressor.jl` is installed in your global `Julia` environment.
   This can be done by activating the `Pkg` mode in the `REPL` by typing `]`.
   This should change the `REPL` to say
```julia
(@v1.x) >
```
where `x` is the version of Julia running in your computer. Next, you simply
type
```julia
add Suppressor
```
and that should install the package in your global environment.

Now, every time you launch Julia from a directory within a `git` repository, you
can locate the `root` directory by typing
```julia
git_root()
```

Furthermore, the script contains a `try/catch` statement that automatically
activates an environment within a `git` repo if available. To confirm this is
working, launch the `Julia REPL` from anywhere in this repo, then enter the
package manager by typing `]`. The `REPL` should look like
```julia
(BayesFitUtils) >
```

Moreover, you can activate the corresponding environment for this project by
typing from anywhere within the `git` repository
```julia
@load_pkg BayesFitUtils
```
This will again change the `Julia REPL` to
```julia
(BayesFitUtils) >
```

Follow the instructions in the main
[`README.md`](https://github.com/mrazomej/bayesian_fitness) file for how to
install the package dependencies for this repository.