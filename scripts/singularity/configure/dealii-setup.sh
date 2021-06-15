cmake \
    -D CMAKE_CXX_COMPILER="mpic++" \
    -D CMAKE_CXX_FLAGS="-march=native -Wno-array-bounds -std=c++17" \
    -D CMAKE_BUILD_TYPE="Release" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D CMAKE_C_COMPILER="mpicc" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D DEAL_II_WITH_METIS="ON" \
    -D P4EST_DIR="/external_libs/p4est-build" \
    -D DEAL_II_WITH_TRILINOS="ON" \
    -D TRILINOS_DIR="/external_libs/trilinos-install" \
    -D DEAL_II_WITH_64BIT_INDICES="OFF" \
    ../dealii
