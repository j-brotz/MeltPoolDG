mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES
   1
   TEST_SOURCE_FILES
   obstacle_forces.cpp
   contact_forces.cpp
   cohesive_forces.cpp
)

mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES
   8
   TEST_SOURCE_FILES
   data_structure.cpp
)
