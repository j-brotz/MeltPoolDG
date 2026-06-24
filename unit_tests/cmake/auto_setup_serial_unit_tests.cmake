# Creates one serial unit test executable per source file using GoogleTest.
# Each source file is compiled into its own executable, linked against
# GTest, GMock, deal.II, and the meltpooldg library, and automatically
# registered with CTest.
#
# Example:
#   mpdg_setup_serial_unit_tests(
#     TEST_SOURCE_FILES
#       test_1.cpp
#       test_2.cpp
#   )
function(mpdg_setup_serial_unit_tests)
    set(OPTIONS "")
    set(ONE_VALUE_ARGS "")
    set(MULTI_VALUE_ARGS TEST_SOURCE_FILES)
    set(ARG_PREFIX "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${ARG_PREFIX}" "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}")

    # Check if file list is not empty assuming that the user might not want to call this function with an empty list
    if(NOT ARG_TEST_SOURCE_FILES)
        message(WARNING "An empty list of test source files was provided to mpdg_setup_serial_unit_tests. This call will therefore not create any tests and is obsolete.")
    endif()

    foreach(TEST_SRC ${ARG_TEST_SOURCE_FILES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME)
        string(REPLACE ".cpp" "" TEST_TARGET ${TEST_NAME})

        add_executable(${TEST_TARGET} ${TEST_SRC})
        target_link_libraries(${TEST_TARGET} GTest::gtest_main GTest::gmock)
        deal_ii_setup_target(${TEST_TARGET})
        target_link_libraries(${TEST_TARGET} ${MELTPOOLDG_LIB})

        add_test(
            NAME ${TEST_TARGET}
            COMMAND ${TEST_TARGET} --gtest_output=xml:unittest_reports/${TEST_TARGET}_report.xml
        )

        set_property(TEST ${TEST_TARGET} PROPERTY LABELS UnitTest Serial)
    endforeach()
endfunction()
