mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES
   1
   TEST_SOURCE_FILES
   obstacle_forces_test.cpp
   contact_forces.cpp
   cohesive_forces.cpp
   dem_time_integrators.cpp
)

mpdg_setup_mpi_unit_tests(
   N_MPI_PROCESSES
   2
   TEST_SOURCE_FILES
   obstacle_data_structure.cpp
)
