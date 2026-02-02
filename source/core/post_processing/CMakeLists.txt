file(GLOB library_source_files "*.cpp")
set(mpdg_post_processing_library_name "mpdg_post_processing")

add_mpdg_library(
    LIBRARY_NAME ${mpdg_post_processing_library_name}
    LIBRARY_SOURCE_FILES ${library_source_files}
)