function(add_mpdg_library)
    set(options "")
    set(one_value_args LIBRARY_NAME)
    set(multi_value_args LIBRARY_SOURCE_FILES LIBRARY_LINK_LIBRARIES)
    set(arg_prefix "ARG")
    cmake_parse_arguments(PARSE_ARGV 0 "${arg_prefix}" "${options}" "${one_value_args}" "${multi_value_args}")

    if(NOT ARG_LIBRARY_NAME)
        message(FATAL_ERROR "LIBRARY_NAME must be provided to add_library.")
    endif()

    if(NOT ARG_LIBRARY_SOURCE_FILES)
        message(FATAL_ERROR "LIBRARY_SOURCE_FILES must be provided to add_library.")
    endif()

    add_library(${ARG_LIBRARY_NAME} ${ARG_LIBRARY_SOURCE_FILES})
    deal_ii_setup_target(${ARG_LIBRARY_NAME})
    target_link_libraries(${meltpooldg_lib} PUBLIC ${ARG_LIBRARY_NAME} ${ARG_LIBRARY_LINK_LIBRARIES})
endfunction()
