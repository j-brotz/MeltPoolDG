# dir to MeltPoolDG dependening on the location of this file
mpDir=$(dirname -- "$0")/../..
mpDir=$(realpath "$mpDir")

read -p "Enter the path where external depencies should be installed (press enter for default=$mpDir/../external_libs): " depDir
depDir=${depDir:-$mpDir/../external_libs}
depDir=$(realpath "$depDir")

read -p 'Enter the number of processes (press enter for default=4): ' np
np=${np:-4}

##############################################################
# install dependencies
##############################################################
mkdir -p $depDir
cd $depDir
bash $mpDir/scripts/config/download_and_install_dependencies.sh $np

###############################################################
## install MeltPoolDG
###############################################################
dealii_dir=$depDir/dealii-build
adaflo_include=$depDir/adaflo/include
adaflo_dir=$depDir/adaflo/build_release

$mpDir/scripts/config/install_meltpooldg.sh $np $dealii_dir $adaflo_include $adaflo_dir
