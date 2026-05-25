function(mpdg_auto_setup_benchmarks)
    file(GLOB BENCHMARK_EXECUTABLES "*.cpp")
    foreach(benchmark_file ${BENCHMARK_EXECUTABLES})
        # Extract file name without path
        get_filename_component(benchmark_name ${benchmark_file} NAME)
        # Remove file extension
        string(REPLACE ".cpp" "" benchmark_exec ${benchmark_name})
        add_executable(${benchmark_exec} ${benchmark_file})
        deal_ii_setup_target(${benchmark_exec})
        target_include_directories(${benchmark_exec} PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks)
        target_link_libraries(${benchmark_exec} ${meltpooldg_lib} ${BENCHMARK_LIBRARY} benchmark::benchmark)
        target_compile_definitions(${benchmark_exec}
            PRIVATE MPDG_BENCHMARK_DATA_DIR="${CMAKE_SOURCE_DIR}/benchmarks/test_data")
    endforeach()
endfunction()
