mpdg_setup_serial_unit_tests(
   TEST_SOURCE_FILES obstacle_forces_test.cpp
)

mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES 4
   TEST_SOURCE_FILES obstacle_data_structure_unit_test.cpp
)