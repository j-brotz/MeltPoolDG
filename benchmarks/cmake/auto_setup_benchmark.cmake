function(mpdg_auto_setup_benchmarks)
    file(GLOB BENCHMARK_EXECUTABLES "*.cpp")
    foreach (BENCHMARK_FILE ${BENCHMARK_EXECUTABLES})
        # Extract file name without path
        get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME)
        # Remove file extension
        string(REPLACE ".cpp" "" BENCHMARK_EXEC ${BENCHMARK_NAME})
        add_executable(${BENCHMARK_EXEC} ${BENCHMARK_FILE})
        deal_ii_setup_target(${BENCHMARK_EXEC})
        target_include_directories(${BENCHMARK_EXEC} PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks)
        target_link_libraries(${BENCHMARK_EXEC} ${MELTPOOLDG_LIB} ${BENCHMARK_LIBRARY} benchmark::benchmark)
    endforeach ()
endfunction()