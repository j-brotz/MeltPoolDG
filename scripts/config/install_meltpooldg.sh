np=${1:-4}
dealii_dir=${2:-../external_libs/dealii-build}
adaflo_include=${3:-../external_libs/adaflo/include}
adaflo_dir=${4:-../external_libs/adaflo/build_release}

# dir to MeltPoolDG dependening on the location of this file
mpDir=$(dirname -- "$0")/../..
##############################################################
# install MeltPoolDG
##############################################################
cd $mpDir
mkdir -p build_release
cd build_release
$mpDir/scripts/config/meltpooldg-config.sh $dealii_dir $adaflo_include $adaflo_dir
make -j$np
