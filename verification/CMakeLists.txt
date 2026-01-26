file(GLOB verification_test_cmake_commands "cmake/*.cmake")
foreach(command ${verification_test_cmake_commands})
    include(${command})
endforeach()

# In order to run tests with MPI, we need to know the MPI executable and its semantics.
find_package(MPI REQUIRED)

add_subdirectory(particles)
