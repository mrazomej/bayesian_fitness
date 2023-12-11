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

# Try to load environment if available within git repo
try
    # Walk through git root directory iteratively
    for (root, dirs, files) in walkdir(git_root())
        # Loop through files. Note: This automatically updates the corresponding
        # root for each file as well without having to explicitly loop through
        # them.
        for file in files
            # Locate Manifest.toml file
            if occursin("Manifest.toml", file)
                # Activate package where Manifest.toml file is found
                Pkg.activate(root)
                # End for loop to make sure only one iteration happens.
                break
            end
        end #for
    end # for
catch
    # Do nothing
end # try/catch