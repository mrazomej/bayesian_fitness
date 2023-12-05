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
