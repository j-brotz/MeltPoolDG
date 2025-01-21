#!/bin/bash

# Determine the directory of the MeltPoolDG script (relative to this script)
mpDir=$(dirname -- "$0")/../..
mpDir=$(realpath "$mpDir")

# Set default directory for dependencies
depDir="$mpDir/../external_libs"

# Get the number of CPU cores available on the machine
core_count=$(nproc)

# Default to one less than the number of CPU cores for parallel processes
np=$((core_count - 1))

# Flag to skip prompts if '--nv' argument is provided
skip_prompts=false

# Parse command line arguments
for arg in "$@"; do
  if [[ "$arg" == "--nv" ]]; then
    skip_prompts=true
  fi
done

# If prompts are not skipped, ask the user for the dependency installation path and number of processes
if [ "$skip_prompts" == false ]; then
  # Prompt for the installation path for external dependencies
  read -p "Enter the path where external dependencies should be installed (default: $depDir): " userDepDir
  depDir=${userDepDir:-$depDir}

  # Prompt for the number of processes for installation
  read -p "Enter the number of processes (default: $np): " user_np
  np=${user_np:-$np}
fi

# Output the paths and number of processes being used
echo "External libraries will be installed to: $depDir"
echo "Number of processes to be used for installation: $np"

# Resolve the absolute path of the dependency directory
depDir=$(realpath "$depDir")

##############################################################
# install dependencies
##############################################################
mkdir -p $depDir
cd "$depDir" || { echo "Failed to change directory to $depDir"; exit 1; }
bash $mpDir/scripts/config/download_and_install_dependencies.sh $np || { echo "Failed to install dependencies."; exit 1; }

###############################################################
## install MeltPoolDG
###############################################################
# Set up directories for MeltPoolDG installation
dealii_dir=$depDir/dealii-build
adaflo_include=$depDir/adaflo/include
adaflo_dir=$depDir/adaflo/build_release

# Install MeltPoolDG
bash $mpDir/scripts/config/install_meltpooldg.sh $np $dealii_dir $adaflo_include $adaflo_dir || { echo "MeltPoolDG installation failed."; exit 1; }

echo "MeltPoolDG installation completed successfully!"
