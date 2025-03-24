#!/bin/bash

# Get the absolute path of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the directory where the script should be executed from (two levels above)
expected_dir="$(cd "$script_dir/../.." && pwd)"

# Get the current working directory
current_dir="$(pwd)"

# Compare the directories
if [[ "$current_dir" == "$expected_dir" ]]; then
    echo "Script is being executed from the correct directory: $current_dir"
else
    echo "Error: Script must be run from $expected_dir, but it is running from $current_dir"
    exit 1
fi

# Exit on error
set -e

# Check for required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <build_folder> <num_processes>"
  exit 1
fi

BUILD_FOLDER="$1"
NUM_PROCESSES="$2"

echo "begin to run clang-tidy"
run-clang-tidy -quiet -p ${BUILD_FOLDER} -j ${NUM_PROCESSES} -use-color  2>${BUILD_FOLDER}/clang-tidy-error.txt >${BUILD_FOLDER}/clang-tidy-detailed.log

# grep interesting errors and make sure we remove duplicates:
grep -E '(warning|error): ' clang-tidy-output.txt | sort | uniq >${BUILD_FOLDER}/clang-tidy.log

echo "clang-tidy completed. See clang-tidy-detailed.log and clang-tidy-error.txt for results."
