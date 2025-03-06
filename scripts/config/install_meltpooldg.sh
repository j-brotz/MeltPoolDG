#!/bin/bash

# Check if help flag is provided
if [[ "$1" == "--help" ]]; then
  echo "Usage: $0 [num_processes] [dealii_dir] [adaflo_include] [adaflo_dir] [buildConfig]"
  echo
  echo "Arguments:"
  echo "  num_processes   (default: 4)      Number of parallel processes for make."
  echo "  dealii_dir      (default: ../external_libs/dealii-build) Path to deal.II build directory."
  echo "  adaflo_include  (default: ../external_libs/adaflo/include) Path to Adaflo include directory."
  echo "  adaflo_dir      (default: ../external_libs/adaflo/build_release) Path to Adaflo build directory."
  echo "  buildConfig     (default: DebugRelease) Build configuration (Release, Debug, or DebugRelease)."
  echo "  pathToBuild     (default: .) Path to directory where the build folders of MeltPoolDG should be located."
  echo
  echo "Example:"
  echo "  $0 8 /path/to/dealii /path/to/adaflo/include /path/to/adaflo/build Release ."
  exit 0
fi

np=${1:-4}
dealii_dir=${2:-../external_libs/dealii-build}
adaflo_include=${3:-../external_libs/adaflo/include}
# TODO: unify to incorporate only path to build folder
adaflo_dir=${4:-../external_libs/adaflo/build_release}
buildConfig=${5:-DebugRelease}
pathToBuild=${6:-.}

# dir to MeltPoolDG dependening on the location of this file
mpDir=$(dirname -- "$0")/../..
mpDir=$(realpath "$mpDir")

##############################################################
# install MeltPoolDG
##############################################################
cd $mpDir
if [[ "$buildConfig" == "Release" || "$buildConfig" == "DebugRelease" ]]; then
  mkdir -p $pathToBuild/build_release
  cd build_release
  $mpDir/scripts/config/meltpooldg-config.sh $dealii_dir $adaflo_include $adaflo_dir Release $pathToBuild
  make -j$np
  cd ..
fi
if [[ "$buildConfig" == "Debug" || "$buildConfig" == "DebugRelease" ]]; then
  mkdir -p $pathToBuild/build_debug
  cd build_debug
  $mpDir/scripts/config/meltpooldg-config.sh $dealii_dir $adaflo_include $adaflo_dir Debug $pathToBuild
  make -j$np
  cd ..
fi
