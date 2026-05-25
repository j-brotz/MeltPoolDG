file(GLOB custom_benchmark_cmake_commands "cmake/*.cmake")
foreach(command ${custom_benchmark_cmake_commands})
    include(${command})
endforeach()

# ##############################################################################
# benchmark library (utilities)
# ##############################################################################
set(BENCHMARK_LIBRARY "mpdg_benchmark_library")
file(GLOB BENCHMARK_SOURCES "${google_benchmark_SOURCE_DIR}/src/*.cc")
# needs to be removed as it is already a part of benchmark::benchmark
list(REMOVE_ITEM BENCHMARK_SOURCES "${google_benchmark_SOURCE_DIR}/src/benchmark_main.cc")
add_library(${BENCHMARK_LIBRARY} benchmark_util.cpp ${BENCHMARK_SOURCES})
deal_ii_setup_target(${BENCHMARK_LIBRARY})
target_link_libraries(${BENCHMARK_LIBRARY} benchmark::benchmark)

# ##############################################################################
# copy benchmark data files to the binary directory
# ##############################################################################
file(GLOB PARTICLE_CSV_FILES "${CMAKE_SOURCE_DIR}/benchmarks/test_data/*")
foreach(SOURCE_FILE IN LISTS PARTICLE_CSV_FILES)
    get_filename_component(FILE_NAME "${SOURCE_FILE}" NAME)
    set(DEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_data/${FILE_NAME}")

    configure_file(
        ${SOURCE_FILE}
        ${DEST_FILE}
        COPYONLY
    )
endforeach()

# ##############################################################################
# add benchmark executables
# ##############################################################################
add_subdirectory(compressible_flow)
add_subdirectory(particles)
