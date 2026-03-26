# Configures an application by checking if a specific option is enabled, collecting source files, and setting up the executable target.
#
# The function takes the following arguments:
# - OUT_APP_ENABLED: A variable to store whether the application is enabled or not.
# - CONFIGURE_OPTION_NAME: The name of the CMake option that controls whether the application is enabled.
# - APPLICATION_NAME: A human-readable name for the application, used in status messages.
# - DESCRIPTION: A description of the application, used in the option description.
function(setup_application OUT_APP_ENABLED)
    set(options "")
    set(one_value_args CONFIGURE_OPTION_NAME APPLICATION_NAME DESCRIPTION)
    set(multi_value_args "")
    set(arg_prefix "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${arg_prefix}" "${options}" "${one_value_args}" "${multi_value_args}")

    option(${ARG_CONFIGURE_OPTION_NAME} "${ARG_DESCRIPTION}" ON)

    if(NOT ${ARG_CONFIGURE_OPTION_NAME})
        message(STATUS "Application: ${ARG_APPLICATION_NAME} disabled")
        set(${OUT_APP_ENABLED} FALSE PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Application: ${ARG_APPLICATION_NAME} enabled")
    set(${OUT_APP_ENABLED} TRUE PARENT_SCOPE)

    # Collect all .cpp files in the current directory
    file(GLOB SOURCE_FILES "*.cpp")

    # Check the number of matching files
    list(LENGTH SOURCE_FILES NUM_FILES)

    if(NUM_FILES EQUAL 1)
        # do nothing
    elseif(NUM_FILES GREATER 1)
        message(FATAL_ERROR "Error: More than one file matches '*.cpp'. Found: ${SOURCE_FILES}")
    else()
        message(FATAL_ERROR "Error: No file matches '*.cpp'.")
    endif()

    # Recursively collect all .cpp files in subdirectories
    file(GLOB_RECURSE CASES "**/*.cpp")

    # Recursively collect all .hpp files in subdirectories
    file(GLOB_RECURSE CASES_HEADERS "**/*.hpp")

    # Loop over each source file found in SOURCE_FILES
    foreach(source_file ${SOURCE_FILES})
        # Extract the name of the executable from the current directory
        get_filename_component(exec ${CMAKE_CURRENT_SOURCE_DIR} NAME)

        # Notify about the executable being created
        message("-- Building executable: ${exec}")

        # Add an executable target with the current source file, all cases, and headers
        add_executable(${exec} ${source_file} ${CASES} ${CASES_HEADERS})

        deal_ii_setup_target(${exec})

        # Link additional libraries (meltpooldg_lib is assumed to be defined elsewhere)
        target_link_libraries(${exec} ${meltpooldg_lib})
    endforeach(source_file ${SOURCE_FILES})
endfunction()
