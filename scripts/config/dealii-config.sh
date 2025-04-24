pwd
trilinos_dir=${1:-../trilinos-install/}
p4est_dir=${2:-../p4est-build}
arborx_dir=${3:-../ArborX-install}
dealii_dir=${4:-../dealii}
buildConfig=${5:-DebugRelease}

rm -rf CMakeFiles/ CMakeCache.txt || true
cmake \
    -D CMAKE_CXX_COMPILER="mpicxx" \
    -D CMAKE_CXX_FLAGS="-march=native -Wno-array-bounds" \
    -D CMAKE_BUILD_TYPE="$buildConfig" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D CMAKE_CXX_STANDARD="20" \
    -D CMAKE_C_COMPILER="mpicc" \
    -D MPIEXEC_PREFLAGS="-bind-to none" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D DEAL_II_WITH_PETSC="OFF" \
    -D DEAL_II_WITH_METIS="ON" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_TRILINOS="ON" \
    -D DEAL_II_WITH_KOKKOS="ON" \
    -D DEAL_II_WITH_ARBORX="ON" \
    -D ARBORX_DIR="$arborx_dir" \
    -D ARBORX_INSTALL_INCLUDE_DIR="$arborx_dir/include" \
    -D TRILINOS_DIR="$trilinos_dir" \
    -D P4EST_DIR="$p4est_dir" \
    -D DEAL_II_WITH_HDF5="OFF" \
    -D DEAL_II_WITH_64BIT_INDICES="OFF" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    $dealii_dir
