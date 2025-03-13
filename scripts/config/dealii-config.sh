pwd
trilinos_install=${1:-../trilinos-install/}
p4est_dir=${2:-../p4est-build}
dealii_dir=${3:-../dealii}
buildConfig=${4:-DebugRelease}

rm -rf CMakeFiles/ CMakeCache.txt || true
cmake \
    -D CMAKE_CXX_COMPILER="mpicxx" \
    -D CMAKE_CXX_FLAGS="-march=native -Wno-array-bounds -std=c++2a" \
    -D CMAKE_BUILD_TYPE="$buildConfig" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D CMAKE_C_COMPILER="mpicc" \
    -D MPIEXEC_PREFLAGS="-bind-to none" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D DEAL_II_WITH_PETSC="OFF" \
    -D DEAL_II_WITH_METIS="ON" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_TRILINOS="ON" \
    -D DEAL_II_WITH_KOKKOS="OFF" \
    -D TRILINOS_DIR=$trilinos_install \
    -D P4EST_DIR=$p4est_dir \
    -D DEAL_II_WITH_HDF5="OFF" \
    -D DEAL_II_WITH_64BIT_INDICES="OFF" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    $dealii_dir
