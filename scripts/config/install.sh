#!/bin/bash

# dry run
#make() {
    #echo "Dummy make command: make $* (Ignoring actual execution)"
#}

#!/bin/bash

# Get the absolute path of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the directory where the script should be executed from (two levels above)
expected_dir="$(cd "$script_dir/../.." && pwd)"

# Get the current working directory
current_dir="$(pwd)"

source $script_dir/log.sh

rm -f $script_dir/../../install.log

# Compare the directories
if [[ "$current_dir" == "$expected_dir" ]]; then
    log "Script is being executed from the correct directory: $current_dir"
else
    log "Error: Script must be run from $expected_dir, but it is running from $current_dir"
    exit 1
fi

# Determine the directory of the MeltPoolDG script (relative to this script)
mpDir=$(dirname -- "$0")/../..
mpDir=$(realpath "$mpDir")

# Set default directory for dependencies
depDir="$mpDir/../external_libs"

# Get the number of CPU cores available on the machine
core_count=$(nproc)

# Default to one less than the number of CPU cores for parallel processes
np=$((core_count - 1))

# Default build configuration
buildConfig="DebugRelease"

# Default path to build folders
buildPath="."

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

  # Prompt for build configuration
  read -p "Select build configuration (Debug/Release/DebugRelease) [Default: DebugRelease]: " buildConfig
  buildConfig=${buildConfig:-DebugRelease}

  if [[ "$buildConfig" != "Debug" && "$buildConfig" != "Release" && "$buildConfig" != "DebugRelease" ]]; then
    log "ERROR: Invalid build configuration. Please choose 'Debug', 'Release' or 'DebugRelease'."
    exit 1
  fi
  log "Selected build configuration: $buildConfig"

  read -p "Select path to location of MeltPoolDG build folders. Make sure that the path exists. [Default: .]: " buildPath
  buildPath=${buildPath:-.}

fi

# Output the paths and number of processes being used
log "External libraries will be installed to: $depDir"
log "Number of processes to be used for installation: $np"
log "Path to build folders: $buildPath"

# Resolve the absolute path of the dependency directory
depDir=$(realpath "$depDir")

##############################################################
# install dependencies
##############################################################
mkdir -p $depDir
cd "$depDir" || { log "Failed to change directory to $depDir"; exit 1; }
bash $mpDir/scripts/config/download_and_install_dependencies.sh $np $buildConfig || { log "ERROR: Failed to install dependencies."; exit 1; }

###############################################################
## install MeltPoolDG
###############################################################
# Set up directories for MeltPoolDG installation
dealii_dir=$depDir/dealii-build
adaflo_include=$depDir/adaflo/include

# Install MeltPoolDG release configuration
if [[ "$buildConfig" == "Release" || "$buildConfig" == "DebugRelease" ]]; then
  adaflo_dir=$depDir/adaflo/build_release
  bash $mpDir/scripts/config/install_meltpooldg.sh $np $dealii_dir $adaflo_include $adaflo_dir Release $pathToBuild || { log "ERROR: MeltPoolDG installation failed."; exit 1; }
fi

# Install MeltPoolDG debug configuration
if [[ "$buildConfig" == "Debug" || "$buildConfig" == "DebugRelease" ]]; then
  adaflo_dir=$depDir/adaflo/build_debug
  bash $mpDir/scripts/config/install_meltpooldg.sh $np $dealii_dir $adaflo_include $adaflo_dir Debug $pathToBuild || { log "ERROR: MeltPoolDG installation failed."; exit 1; }
fi

log "MeltPoolDG installation completed successfully!"
