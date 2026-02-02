file(GLOB_RECURSE physics_sources
    "${CMAKE_SOURCE_DIR}/source/physics/**/*.cpp")

set(PHYSICS_LIBRARY_NAME "mpdg_physics_library")
add_library(${PHYSICS_LIBRARY_NAME} OBJECT ${physics_sources})
target_link_libraries(${PHYSICS_LIBRARY_NAME} PUBLIC ${CORE_LIBRARY_NAME} ${NUMERICS_LIBRARY_NAME})
target_link_libraries(${meltpooldg_lib} INTERFACE ${PHYSICS_LIBRARY_NAME})