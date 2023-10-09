# Bayesian inference of relative fitness on high-throughput pooled competition assays

Welcome to the GitHub repository for the bayesian fitness project! This
repository serves as a record for the theoretical and computational work
described in the publication "[`TBD`]" 

## Branches
[`include information about GitHub repository branches`]

## Installation
The installation of the `BayesFitUtils.jl` package requires `Julia` version `>
1.9.0`. We direct the user to the [Julia language
homepage](https://julialang.org) for instructions on how to install the latest
version of Julia.

**Note:** This package was originally developed using `Julia v1.9.3`.

### Installing the `BayesFitUtils` package
To maintain a clean separate environment for this project, we created a package
named `BayesFitUtils` with the custom functions needed to reproduce our
analysis. This package **does not contain** the functions needed to run the 
Bayesian inference pipeline. Such functions are part of an independent package
[BayesFitness.jl](https://github.com/mrazomej/BayesFitness.jl). However,
`BayesFitUtils` has `BayesFitness.jl` as part of its dependencies. To install
the `BayesFitUtils` package, follow the following steps:

1. From the terminal, navigate to the `BayesFitUtils` folder.

2. Launch `Julia` by typing the command
```
julia
```

3. Activate the `Pkg` mode in the `REPL` by typing `]`. This should change the
   `REPL` to say
```
(@v1.x) >
```
where `x` is the version of `Julia`. Note: The `REPL` `Pkg` mode is the native
package manager for the `Julia` language.

0. **Extra steps**: Since the `BayesFitness.jl` package is not yet registered,
   this might give an error when trying to install all dependencies for the
   `BayesFitUtils` package. Therefore, add the package directly by typing
```
add https://github.com/mrazomej/BayesFitness.jl
```
This step will be deprecated once the package is registered.

4. Run the command:
```
activate .
```
to activate the `BayesFitUtils` package. The `REPL` should now show
```
(BayesFitUtils) >
```

5. Run the command
```
instantiate
```
This should install all the necessary packages and the dependencies listed in
the `Project.toml` file.

## Optional setup to match package import
Throughout this repository, there are two commands utilized when importing the
packages for each script or `Jupyter` notebook:

- `git_root()`: Function to find the root environment of a particular Git
  repository.

- `load_pkg(pkg)`: Function to find the home directory of a `GitHub` repository
to load the package for that particular project.

These functions are part of the `~/.julia/config/startup.jl` file. To add these
functions to your main Julia installation, follow the instructions in the
`README.md` file in the `miscellaneous/` directory.

## License
![](https://licensebuttons.net/l/by/3.0/88x31.png)

All creative works (writing, figures, etc) are licensed under the [Creative
Commons CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. All
software is distributed under the standard MIT license as follows

```
Copyright 2023 The Authors 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```