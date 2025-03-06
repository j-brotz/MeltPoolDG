dealii_dir=${1:-../dealii-build}
adaflo_include=${2:-../adaflo/include}
adaflo_dir=${3:-../adaflo/build_release}
buildConfig=${4:-Release}


rm -rf CMakeFiles/ CMakeCache.txt || true
cmake \
	-DADAFLO_LIB=$adaflo_dir \
	-DADAFLO_INCLUDE=$adaflo_include \
	-DCMAKE_BUILD_TYPE=$buildConfig \
	-DDEAL_II_DIR=$dealii_dir \
	-DBUILD_SHARED_LIBS=true \
	-DCMAKE_CXX_COMPILER=mpicxx \
	..
