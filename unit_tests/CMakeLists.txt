set(TEST_LIBRARIES ${meltpooldg_lib})
set(TEST_TARGET ${meltpooldg_lib})
deal_ii_pickup_tests()

file(GLOB unit_test_cmake_commands "cmake/*.cmake")
foreach(command ${unit_test_cmake_commands})
    include(${command})
endforeach()

# In order to run tests with MPI, we need to know the MPI executable and its semantics.
find_package(MPI REQUIRED)

add_subdirectory(particles)
