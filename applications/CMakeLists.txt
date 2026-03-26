file(GLOB custom_application_cmake_commands "${CMAKE_SOURCE_DIR}/applications/cmake/*.cmake")
foreach (command ${custom_application_cmake_commands})
    include(${command})
endforeach ()

file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child} AND NOT child STREQUAL "cmake")
        add_subdirectory(${child})
    endif()
endforeach()