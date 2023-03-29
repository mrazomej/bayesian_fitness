## `miscellaneous`

### `startup.jl`
Script containing necessary functions to easily import the project software
package (`BayesFitUtils`) from any directory within the repository.

To add this functionality, follow these steps:
1. Add the `Suppressor.jl` package to your main `Julia` installation by typing
```
add Suppressor
```
from the `Pkg` manager in the `Julia REPL`.

2. copy this script to the directory
```
~/.julia/config/
```