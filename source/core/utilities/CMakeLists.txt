file(GLOB library_source_files "*.cpp")
set(mpdg_utilities_library_name "mpdg_utilities")

add_mpdg_library(
    LIBRARY_NAME ${mpdg_utilities_library_name}
    LIBRARY_SOURCE_FILES ${library_source_files}
)

target_compile_definitions(${mpdg_utilities_library_name} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
