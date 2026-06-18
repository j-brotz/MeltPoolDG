file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child} AND NOT child STREQUAL "cmake")
        add_subdirectory(${child})
    endif()
endforeach()
