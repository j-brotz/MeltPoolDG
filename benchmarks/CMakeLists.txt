file(GLOB custom_benchmark_cmake_commands "cmake/*.cmake")
foreach (command ${custom_benchmark_cmake_commands})
    include(${command})
endforeach ()

add_subdirectory(flow)