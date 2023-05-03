#number of processes
np=${1:-4}

#config dir (MeltPoolDG dependening on the location of this file)
configDir=$(dirname -- "$0")
configDir=$(realpath "$configDir")

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
wget http://p4est.github.io/release/p4est-2.0.tar.gz
$configDir/p4est-config.sh p4est-2.0.tar.gz `pwd/p4est-build`

##############################################################
# install Trilinos
##############################################################
wget https://github.com/trilinos/Trilinos/archive/trilinos-release-13-4-1.tar.gz
mv trilinos-release-13-4-1.tar.gz trilinos-release-13-4-1.tar
tar -xvf trilinos-release-13-4-1.tar
mkdir -p trilinos-build
cd trilinos-build
rm -rf *
$configDir/trilinos-config.sh
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

