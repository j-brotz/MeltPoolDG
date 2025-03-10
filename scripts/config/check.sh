#!/bin/bash

# Get the absolute path of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/log.sh

##############################################################
# check proper cmake version
##############################################################
check_cmake_version() {
  cmp=3.17.0
  ver=$(cmake --version | head -1 | cut -f3 -d" ")

  mapfile -t sorted < <(printf "%s\n" "$ver" "$cmp" | sort -V)

  if [[ ${sorted[0]} == "$cmp" ]]; then
      log "cmake version $ver >= $cmp"
  else
      log "ERROR: cmake version too low; update to at least $cmp."
      exit 1
  fi
}

##############################################################
# check if metis is installed
##############################################################
check_metis() {
  ldconfig -p | grep libmetis

  if [[ $(ldconfig -p | grep libmetis) ]]; then
      log "Dependency libmetis found."
  else
      log "WARNING: Dependency libmetis not found. Make sure to install metis if you want to use the functionalities."
      exit 1
  fi
}

check_metis # Call function

##############################################################
# check if boost is installed
##############################################################
# Function to check if Boost is installed
check_boost() {
    if ldconfig -p | grep -q libboost; then
        log "Boost is installed."
    elif [[ -f "/usr/include/boost/version.hpp" || -f "/usr/local/include/boost/version.hpp" ]]; then
        log "Boost headers found."
    elif pkg-config --exists boost; then
        log "Boost detected via pkg-config."
    else
        log "ERROR: Boost library is missing. Install Boost before proceeding."
        exit 1
    fi
}

check_boost # Call function

##############################################################
# check if blas is installed
##############################################################
check_blas() {
    if ldconfig -p | grep -E 'libblas\.so|libopenblas\.so' > /dev/null; then
        log "BLAS is installed."
    else
        log "ERROR: BLAS is not installed. Please install BLAS before proceeding."
        exit 1
    fi
}

##############################################################
# check if openMPI is installed and the version is correct
##############################################################

check_mpi() {
  # Get the MPI version (assumes OpenMPI)
  mpi_version=$(mpirun --version 2>/dev/null | head -n 1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+(-[0-9]+)?')

  # Required version
  required_version="4.1.2-2"

  # Function to normalize versions (convert to numeric comparison-friendly format)
  normalize_version() {
      echo "$1" | awk -F '[-.]' '{ printf("%d%03d%03d%03d%03d\n", $1, $2, $3, ($4 ? $4 : 0), ($5 ? $5 : 0)) }'
  }

  # Check if MPI is installed
  if [[ -z "$mpi_version" ]]; then
      log "MPI is not installed or mpirun is not found."
      exit 1
  fi

  # Normalize versions for comparison
  mpi_version_norm=$(normalize_version "$mpi_version")
  required_version_norm=$(normalize_version "$required_version")

  # Compare versions
  if [[ "$mpi_version_norm" -gt "$required_version_norm" ]]; then
      log "MPI version ($mpi_version) is greater than $required_version."
  else
      log "MPI version ($mpi_version) is NOT greater than $required_version."
      exit 1
  fi
}


##############################################################
# check all
##############################################################
check_all() {
  check_cmake_version
  check_metis
  check_boost
  check_blas
  check_mpi
}
