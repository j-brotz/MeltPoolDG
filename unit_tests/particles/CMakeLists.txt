mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES 1
   TEST_SOURCE_FILES obstacle_forces_test.cpp contact_forces.cpp
)

mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES 4
   TEST_SOURCE_FILES obstacle_data_structure.cpp
)
