# Configures an application by checking if a specific option is enabled, collecting source files, and setting up the executable target.
#
# The function takes the following arguments:
# - OUT_APP_ENABLED: A variable to store whether the application is enabled or not.
# - CONFIGURE_OPTION_NAME: The name of the CMake option that controls whether the application is enabled.
# - APPLICATION_NAME: A human-readable name for the application, used in status messages.
# - DESCRIPTION: A description of the application, used in the option description.
# - DEPENDENCIES: A list of boolean CMake variables that must evaluate to TRUE.
function(setup_application OUT_APP_ENABLED)
    set(OPTIONS "")
    set(ONE_VALUE_ARGS CONFIGURE_OPTION_NAME APPLICATION_NAME DESCRIPTION)
    set(MULTI_VALUE_ARGS DEPENDENCIES)
    set(ARG_PREFIX "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${ARG_PREFIX}" "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}")

    # Check whether all dependencies are available
    set(ALL_DEPENDENCIES_FOUND TRUE)
    set(MISSING_DEPENDENCIES "")

    foreach(DEP IN LISTS ARG_DEPENDENCIES)
        if(NOT DEFINED ${DEP} OR NOT ${${DEP}})
            set(ALL_DEPENDENCIES_FOUND FALSE)
            list(APPEND MISSING_DEPENDENCIES ${DEP})
        endif()
    endforeach()

    # Detect whether the option was explicitly provided before calling this function
    if(DEFINED ${ARG_CONFIGURE_OPTION_NAME})
        set(APP_OPTION_WAS_EXPLICITLY_SET TRUE)
    else()
        set(APP_OPTION_WAS_EXPLICITLY_SET FALSE)
    endif()

    # If the option was not explicitly set, choose the default based on dependencies
    if(NOT APP_OPTION_WAS_EXPLICITLY_SET)
        if(ALL_DEPENDENCIES_FOUND)
            set(DEFAULT_VALUE ON)
        else()
            set(DEFAULT_VALUE OFF)
        endif()
        option(${ARG_CONFIGURE_OPTION_NAME} "${ARG_DESCRIPTION}" ${DEFAULT_VALUE})
    else()
        # Preserve the user-provided cache value
        option(${ARG_CONFIGURE_OPTION_NAME} "${ARG_DESCRIPTION}" "${${ARG_CONFIGURE_OPTION_NAME}}")
    endif()

    # If explicitly enabled, missing dependencies are a hard error
    if(${ARG_CONFIGURE_OPTION_NAME} AND NOT ALL_DEPENDENCIES_FOUND)
        if(APP_OPTION_WAS_EXPLICITLY_SET)
            message(FATAL_ERROR
                "Application '${ARG_APPLICATION_NAME}' was explicitly enabled "
                "(${ARG_CONFIGURE_OPTION_NAME}=ON), but required dependencies are missing: "
                "${MISSING_DEPENDENCIES}")
        else()
            message(STATUS
                "Application: ${ARG_APPLICATION_NAME} disabled "
                "(missing dependencies: ${MISSING_DEPENDENCIES})")
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
    foreach(SOURCE_FILE ${SOURCE_FILES})
        # Extract the name of the executable from the current directory
        get_filename_component(EXEC ${CMAKE_CURRENT_SOURCE_DIR} NAME)

        # Notify about the executable being created
        message("-- Building executable: ${EXEC}")

        # Add an executable target with the current source file, all cases, and headers
        add_executable(${EXEC} ${SOURCE_FILE} ${CASES} ${CASES_HEADERS})

        deal_ii_setup_target(${EXEC})

        # Link additional libraries (MELTPOOLDG_LIB is assumed to be defined elsewhere)
        target_link_libraries(${EXEC} ${MELTPOOLDG_LIB})
    endforeach(SOURCE_FILE ${SOURCE_FILES})
endfunction()
