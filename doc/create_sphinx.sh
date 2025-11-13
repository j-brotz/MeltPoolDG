#!/bin/bash

# Get the absolute path of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the directory where the script should be executed from (two levels above)
expected_dir="$(cd "$script_dir/.." && pwd)"

# Get the current working directory
current_dir="$(pwd)"

# Compare the directories
if [[ "$current_dir" == "$expected_dir" ]]; then
    echo "Script is being executed from the correct directory: $current_dir"
else
    echo "Error: Script must be run from $expected_dir, but it is running from $current_dir"
    exit 1
fi

doxygen doc/doxygen/Doxyfile
doxysphinx build doc/sphinx/source/ doc/sphinx/_build/ doc/doxygen/Doxyfile
sphinx-build -v -b html -a doc/sphinx/source/ doc/sphinx/_build
