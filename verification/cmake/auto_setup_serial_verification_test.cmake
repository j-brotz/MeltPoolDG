# Creates one serial verification test executable per source file using GoogleTest.
# Each source file is compiled into its own executable, linked against
# GTest, GMock, deal.II, and the meltpooldg library, and automatically
# registered with CTest.
#
# Example:
#   mpdg_setup_serial_verification_tests(
#     TEST_SOURCE_FILES
#       test_1.cpp
#       test_2.cpp
#   )
function(mpdg_setup_serial_verification_tests)
    set(options "")
    set(one_value_args "")
    set(multi_value_args TEST_SOURCE_FILES)
    set(arg_prefix "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${arg_prefix}" "${options}" "${one_value_args}" "${multi_value_args}")

    # Check if file list is not empty assuming that the user might not want to call this function with an empty list
    if(NOT ARG_TEST_SOURCE_FILES)
        message(WARNING "An empty list of test source files was provided to mpdg_setup_serial_verification_tests. This call will therefore not create any tests and is obsolete.")
    endif()

    foreach(test_src ${ARG_TEST_SOURCE_FILES})
        get_filename_component(test_name ${test_src} NAME)
        string(REPLACE ".cpp" "" test_target ${test_name})

        add_executable(${test_target} ${test_src})
        target_link_libraries(${test_target} GTest::gtest_main GTest::gmock)
        deal_ii_setup_target(${test_target})
        target_link_libraries(${test_target} ${meltpooldg_lib})

        add_test(
            NAME ${test_target}
            COMMAND ${test_target} --gtest_output=xml:verification_reports/${test_target}_report.xml
        )

        set_property(TEST ${test_target} PROPERTY LABELS VerificationTest Serial)
    endforeach()
endfunction()
