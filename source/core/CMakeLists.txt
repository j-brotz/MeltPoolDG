file(GLOB_RECURSE core_sources
    "${CMAKE_SOURCE_DIR}/source/core/**/*.cpp")

set(CORE_LIBRARY_NAME "mpdg_core_library")
add_library(${CORE_LIBRARY_NAME} OBJECT ${core_sources})
target_link_libraries(${meltpooldg_lib} INTERFACE ${CORE_LIBRARY_NAME})