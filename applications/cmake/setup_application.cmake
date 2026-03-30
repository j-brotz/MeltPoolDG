# Configures an application by checking if a specific option is enabled, collecting source files, and setting up the executable target.
#
# The function takes the following arguments:
# - OUT_APP_ENABLED: A variable to store whether the application is enabled or not.
# - CONFIGURE_OPTION_NAME: The name of the CMake option that controls whether the application is enabled.
# - APPLICATION_NAME: A human-readable name for the application, used in status messages.
# - DESCRIPTION: A description of the application, used in the option description.
# - DEPENDENCIES: A list of boolean CMake variables that must evaluate to TRUE.
function(setup_application OUT_APP_ENABLED)
    set(options "")
    set(one_value_args CONFIGURE_OPTION_NAME APPLICATION_NAME DESCRIPTION)
    set(multi_value_args DEPENDENCIES)
    set(arg_prefix "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${arg_prefix}" "${options}" "${one_value_args}" "${multi_value_args}")

    # Check whether all dependencies are available
    set(all_dependencies_found TRUE)
    set(missing_dependencies "")

    foreach(dep IN LISTS ARG_DEPENDENCIES)
        if(NOT DEFINED ${dep} OR NOT ${${dep}})
            set(all_dependencies_found FALSE)
            list(APPEND missing_dependencies ${dep})
        endif()
    endforeach()

    # Detect whether the option was explicitly provided before calling this function
    if(DEFINED ${ARG_CONFIGURE_OPTION_NAME})
        set(app_option_was_explicitly_set TRUE)
    else()
        set(app_option_was_explicitly_set FALSE)
    endif()

    # If the option was not explicitly set, choose the default based on dependencies
    if(NOT app_option_was_explicitly_set)
        if(all_dependencies_found)
            set(default_value ON)
        else()
            set(default_value OFF)
        endif()
        option(${ARG_CONFIGURE_OPTION_NAME} "${ARG_DESCRIPTION}" ${default_value})
    else()
        # Preserve the user-provided cache value
        option(${ARG_CONFIGURE_OPTION_NAME} "${ARG_DESCRIPTION}" "${${ARG_CONFIGURE_OPTION_NAME}}")
    endif()

    # If explicitly enabled, missing dependencies are a hard error
    if(${ARG_CONFIGURE_OPTION_NAME} AND NOT all_dependencies_found)
        if(app_option_was_explicitly_set)
            message(FATAL_ERROR
                "Application '${ARG_APPLICATION_NAME}' was explicitly enabled "
                "(${ARG_CONFIGURE_OPTION_NAME}=ON), but required dependencies are missing: "
                "${missing_dependencies}")
        else()
            message(STATUS
                "Application: ${ARG_APPLICATION_NAME} disabled "
                "(missing dependencies: ${missing_dependencies})")
            set(${OUT_APP_ENABLED} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()


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
