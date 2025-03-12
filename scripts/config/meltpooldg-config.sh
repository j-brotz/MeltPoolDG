dealii_dir=${1:-../dealii-build}
adaflo_include=${2:-../adaflo/include}
adaflo_dir=${3:-../adaflo/build_release}
buildConfig=${4:-Release}
mpDir=${5:-..}

  if [[ "$buildConfig" != "Debug" && "$buildConfig" != "Release" ]]; then
    log "ERROR: Invalid build configuration. Please choose 'Debug' or 'Release'."
    exit 1
  fi

rm -rf CMakeFiles/ CMakeCache.txt || true
cmake \
	-DADAFLO_LIB=$adaflo_dir \
	-DADAFLO_INCLUDE=$adaflo_include \
	-DCMAKE_BUILD_TYPE=$buildConfig \
	-DDEAL_II_DIR=$dealii_dir \
	-DBUILD_SHARED_LIBS=true \
	-DCMAKE_CXX_COMPILER=mpicxx \
	$mpDir
