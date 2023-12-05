# `startup.jl`

This directory contains a `Julia` script `startup.jl` that contains two
functions used throughout this repository:

- `git_root()`: Function to find the root environment of a particular Git
  repository.

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

Furthermore, you can activate the corresponding environment for this project by
typing from anywhere within the `git` repository
```julia
@load_pkg BayesFitUtils
```
This will change the `Julia REPL` to
```julia
(BayesFitUtils) >
```

Follow the instructions in the main
[`README.md`](https://github.com/mrazomej/bayesian_fitness) file for how to
install the package dependencies for this repository.

## `startup.jl` script
```julia
# Import package to supress warning
import Suppressor

# Import package manager
# Suppressor.@suppress 
import Pkg

@doc raw"""
    `git_root()`

Function to find the root environment of a particular Git repository
"""
function git_root()
    # Load package for project by locating git home folder
    home_dir = read(`git rev-parse --show-toplevel`, String)
    return home_dir[1:end-1]
end # function

@doc raw"""
    `load_pkg(pkg)`

Function to find the home directory of a github repository to load the package
for that particular project.

# Arguments
- `pkg::String`: Name of the folder within the `GitHub` repository where the
  package is located.
"""
function load_pkg(pkg)
    # Suppress warning
    Suppressor.@suppress begin
        # Save current directory
        dir = read(`pwd`, String)
        dir = dir[1:end-1]

        # Change directories
        cd(git_root() * "/" * String(pkg))

        # Activate environment
        Pkg.pkg"activate ."

        # Return to original directory
        cd(dir)
    end # @suppress

end # function

macro load_pkg(pkg)
    load_pkg(pkg)
    return nothing
end # macro

```