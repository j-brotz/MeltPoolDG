function(add_mpdg_application)
    set(options "")
    set(one_value_args "")
    set(multi_value_args LIBRARY_NAMES)
    set(arg_prefix "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${arg_prefix}" "${options}" "${one_value_args}" "${multi_value_args}")
    
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

    file(GLOB_RECURSE CASES "**/*.cpp")
    file(GLOB_RECURSE CASES_HEADERS "**/*.hpp")

    # Loop over each source file found in SOURCE_FILES
    foreach(source_file ${SOURCE_FILES})
        get_filename_component(exec ${CMAKE_CURRENT_SOURCE_DIR} NAME)

        message("-- Building executable: ${exec}")
        add_executable(${exec} ${source_file} ${CASES} ${CASES_HEADERS})

        deal_ii_setup_target(${exec})

        target_link_libraries(${exec} ${meltpooldg_lib} ${ARG_LIBRARY_NAMES})
    endforeach(source_file ${SOURCE_FILES})
endfunction()
