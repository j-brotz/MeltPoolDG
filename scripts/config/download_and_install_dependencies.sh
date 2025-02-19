#number of processes
np=${1:-4}
buildConfig=${2:"DebugRelease"}

#config dir (MeltPoolDG dependening on the location of this file)
configDir=$(dirname -- "$0")
configDir=$(realpath "$configDir")

# Get the absolute path of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/log.sh

# load and apply checks
source $script_dir/check.sh
check_all

##############################################################
# install p4est
##############################################################
wget http://p4est.github.io/release/p4est-2.2.tar.gz
mkdir -p "$(pwd)/p4est-build"
$configDir/p4est-config.sh p4est-2.2.tar.gz "$(pwd)/p4est-build"
rm p4est-2.2.tar.gz

#############################################################
# install Trilinos
#############################################################
mkdir -p trilinos-build

log "Use Trilinos version 14.4.0."
wget https://github.com/trilinos/Trilinos/archive/trilinos-release-14-4-0.tar.gz
mv trilinos-release-14-4-0.tar.gz trilinos-release-14-4-0.tar
tar -xvf trilinos-release-14-4-0.tar
cd trilinos-build
rm -rf *
$configDir/trilinos-config.sh ../Trilinos-trilinos-release-14-4-0
rm ../trilinos-release-14-4-0.tar

make -j$np install
cd ..

#############################################################
# install Arborx
#############################################################
wget https://github.com/arborx/ArborX/archive/refs/tags/v1.5.tar.gz
mv v1.5.tar.gz v1.5.tar
tar -xvf v1.5.tar
ARBOX_OPTIONS=(
   -D CMAKE_INSTALL_PREFIX="./ArborX-install"
   -D ARBORX_ENABLE_MPI=ON
   -D Kokkos_ROOT="./trilinos-install/lib/cmake/Kokkos/"
   )
cmake "${ARBOX_OPTIONS[@]}" "${ARBORX_DIR:-./ArborX-1.5/}"
make install

##############################################################
# install deal.II
##############################################################
#git clone https://github.com/dealii/dealii
mkdir -p dealii-build
cd dealii-build
rm -rf *
$configDir/dealii-config.sh ../trilinos-install ../p4est-build ../ArborX-install ../dealii $buildConfig
make -j$np
cd ..

#############################################################
# install adaflo
#############################################################
git clone https://github.com/MeltPoolDG/adaflo
# release
cd adaflo
if [[ "$buildConfig" == "Release" || "$buildConfig" == "DebugRelease" ]]; then
  mkdir -p build_release
  cd build_release
  cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
  make -j$np adaflo
  cd ..
fi
# debug
if [[ "$buildConfig" == "Debug" || "$buildConfig" == "DebugRelease" ]]; then
  mkdir -p build_debug
  cd build_debug
  cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
  make -j$np adaflo
  cd ..
fi

log ""
log "Dependencies successfully installed. You may add the folders to your path via"
log "'export PATH=$(realpath $pwd/dealii-build):$PATH' >> ~/.bashrc"
log "'export ADAFLO_INCLUDE=$(realpath $pwd/adaflo/include):$PATH' >> ~/.bashrc"
