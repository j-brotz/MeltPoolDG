# Creates MPI-enabled unit test executables using GoogleTest. Each source file 
# is compiled into its own executable, linked against GTest, GMock, deal.II, 
# and the meltpooldg library, and registered with CTest to be run via mpiexec.
#
# Example:
#   mpdg_setup_mpi_unit_tests(
#     N_MPI_PROCESSES 4
#     TEST_SOURCE_FILES
#       test_1.cpp
#       test_2.cpp
#   )
function(mpdg_setup_mpi_unit_tests)
    set(options "")
    set(one_value_args N_MPI_PROCESSES)
    set(multi_value_args TEST_SOURCE_FILES)
    set(arg_prefix "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${arg_prefix}" "${options}" "${one_value_args}" "${multi_value_args}")

    # Check validity of number of MPI processes
    if(NOT ARG_N_MPI_PROCESSES MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "N_MPI_PROCESSES must be a positive integral value.")
    endif()

    # Check if file list is not empty assuming that the user might not want to call this function with an empty list
    if(NOT ARG_TEST_SOURCE_FILES)
        message(WARNING "An empty list of test source files was provided to mpdg_setup_mpi_unit_tests. This call will therefore not create any tests and is obsolete.")
    endif()

    foreach(test_src ${ARG_TEST_SOURCE_FILES})
        get_filename_component(test_name ${test_src} NAME)
        string(REPLACE ".cpp" "" test_target ${test_name})
        add_executable(${test_target} ${CMAKE_SOURCE_DIR}/unit_tests/test_utils/gtest_mpi_main.cpp ${test_src})

        target_include_directories(${test_target} PRIVATE ${CMAKE_SOURCE_DIR}/unit_tests/test_utils)
        target_link_libraries(${test_target} GTest::gtest GTest::gmock)
        deal_ii_setup_target(${test_target})
        target_link_libraries(${test_target} ${meltpooldg_lib})

        set(mpi_arguments
            ${MPIEXEC_NUMPROC_FLAG}
            1
            $<TARGET_FILE:${test_target}>
            --gtest_output=xml:unittest_reports/${test_target}_report.xml)

        if(ARG_N_MPI_PROCESSES GREATER 1)
            math(EXPR _n_remaining_processes "${ARG_N_MPI_PROCESSES} - 1")
            list(APPEND mpi_arguments ":" ${MPIEXEC_NUMPROC_FLAG} ${_n_remaining_processes} $<TARGET_FILE:${test_target}>)
        endif()

        # We (at least for now) do not use gtest_discover_tests here because it was not designed for MPI
        add_test(
            NAME ${test_target}
            COMMAND ${MPIEXEC_EXECUTABLE} ${mpi_arguments}
        )

        set_property(TEST ${test_target} PROPERTY LABELS UnitTest MPI)
        set_property(TEST ${test_target} PROPERTY PROCESSORS ${ARG_N_MPI_PROCESSES})
    endforeach()
endfunction()
