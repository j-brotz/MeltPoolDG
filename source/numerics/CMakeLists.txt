file(GLOB_RECURSE numerics_sources
    "${CMAKE_SOURCE_DIR}/source/numerics/**/*.cpp")

set(NUMERICS_LIBRARY_NAME "mpdg_numerics_library")
add_library(${NUMERICS_LIBRARY_NAME} OBJECT ${numerics_sources})
target_link_libraries(${NUMERICS_LIBRARY_NAME} PUBLIC ${CORE_LIBRARY_NAME})
target_link_libraries(${meltpooldg_lib} INTERFACE ${NUMERICS_LIBRARY_NAME})