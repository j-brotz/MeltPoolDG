#number of processes
np=${1:-4}

#config dir (MeltPoolDG dependening on the location of this file)
configDir=$(dirname -- "$0")
configDir=$(realpath "$configDir")

##############################################################
# check proper cmake version
##############################################################
cmp=3.17.0
ver=$(cmake --version | head -1 | cut -f3 -d" ")

mapfile -t sorted < <(printf "%s\n" "$ver" "$cmp" | sort -V)

if [[ ${sorted[0]} == "$cmp" ]]; then
    echo "cmake version $ver >= $cmp"
else
    echo "ERROR: cmake version too low; update to at least $cmp."
    exit 1
fi

##############################################################
# check gcc version
##############################################################
# Get the GCC version
gcc_version=$(gcc --version | head -n 1 | awk '{print $3}')

# Extract the major version
gcc_major_version=$(echo $gcc_version | cut -d'.' -f1)

##############################################################
# install metis
##############################################################
ldconfig -p | grep libmetis

if [[ $(ldconfig -p | grep libmetis) ]]; then
    echo "Dependency libmetis found."
else
    echo "WARNING: Dependency libmetis not found. Make sure to install metis if you want to use the functionalities."
fi

##############################################################
# install p4est
##############################################################
wget http://p4est.github.io/release/p4est-2.2.tar.gz
$configDir/p4est-config.sh p4est-2.2.tar.gz `pwd/p4est-build`

##############################################################
# install Trilinos
##############################################################
mkdir -p trilinos-build

if [ "$gcc_major_version" -lt 13 ]; then
  echo "Use patched Trilinos version 13.4.1."
  wget https://github.com/trilinos/Trilinos/archive/trilinos-release-13-4-1.tar.gz
  mv trilinos-release-13-4-1.tar.gz trilinos-release-13-4-1.tar
  tar -xvf trilinos-release-13-4-1.tar
  cd trilinos-build
  rm -rf *
  $configDir/trilinos-config.sh ../Trilinos-trilinos-release-13-4-1
else 
  echo "Use patched Trilinos version 13.4.1 for GCC13"
  wget https://github.com/MeltPoolDG/Trilinos/archive/trilinos-release-13-4-1-for-gcc-13.zip
  unzip trilinos-release-13-4-1-for-gcc-13.zip
  cd trilinos-build
  rm -rf *
  $configDir/trilinos-config.sh ../Trilinos-trilinos-release-13-4-1-for-gcc-13
fi

make -j$np install
cd ..

###############################################################
## install deal.II
###############################################################
git clone https://github.com/dealii/dealii
mkdir -p dealii-build
cd dealii-build
rm -rf *
$configDir/dealii-config.sh
make -j$np
cd ..

##############################################################
# install adaflo
##############################################################
git clone https://github.com/MeltPoolDG/adaflo
# release
cd adaflo
mkdir -p build_release
cd build_release
cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
make -j$np adaflo
cd ..
# debug
mkdir -p build_debug
cd build_debug
cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
make -j$np adaflo
cd ..

echo ""
echo "Dependencies successfully installed. You may set the environment variables:"
echo "DEAL_II_DIR=$(realpath $pwd/dealii-build)"
echo "ADAFLO_INCLUDE=$(realpath $pwd/adaflo/include)"

