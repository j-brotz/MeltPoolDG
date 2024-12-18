trilinos_dir=${1:-../Trilinos-trilinos-release-13-4-1}
trilinos_install=${2:-../trilinos-install}
rm -rf CMakeFiles/ CMakeCache.txt || true
cmake                                            \
    -DTrilinos_ENABLE_Amesos=ON                      \
    -DTrilinos_ENABLE_Epetra=ON                      \
    -DTrilinos_ENABLE_EpetraExt=ON                   \
    -DTrilinos_ENABLE_Ifpack=ON                      \
    -DTrilinos_ENABLE_AztecOO=ON                     \
    -DTrilinos_ENABLE_Sacado=ON                      \
    -DTrilinos_ENABLE_Teuchos=ON                     \
    -DTrilinos_ENABLE_MueLu=ON                       \
    -DTrilinos_ENABLE_ML=ON                          \
    -DTrilinos_ENABLE_ROL=ON                         \
    -DTrilinos_ENABLE_Tpetra=ON                      \
    -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON              \
    -DTrilinos_ENABLE_COMPLEX_FLOAT=ON               \
    -DTrilinos_ENABLE_Zoltan=ON                      \
    -DTrilinos_VERBOSE_CONFIGURE=OFF                 \
    -DTPL_ENABLE_MPI=ON                              \
    -DBUILD_SHARED_LIBS=ON                           \
    -DCMAKE_VERBOSE_MAKEFILE=OFF                     \
    -DCMAKE_BUILD_TYPE=release                       \
    -DCMAKE_CXX_COMPILER="mpicxx"                    \
    -DCMAKE_C_COMPILER="mpicc"                       \
    -DCMAKE_FORTRAN_FLAGS="-march=native -O3" \
    -DCMAKE_C_FLAGS="-march=native -O3" \
    -DCMAKE_CXX_FLAGS="-march=native -O3" \
    -DCMAKE_INSTALL_PREFIX:PATH=$trilinos_install \
    $trilinos_dir
