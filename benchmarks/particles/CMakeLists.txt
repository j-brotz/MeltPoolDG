mpdg_auto_setup_benchmarks()

# 1. Gather all files from the source directory
file(GLOB PARTICLE_CSV_FILES "${CMAKE_SOURCE_DIR}/benchmarks/particles/particle_locations/*")

# 2. Loop through each file and copy it to the binary directory
foreach(SOURCE_FILE IN LISTS PARTICLE_CSV_FILES)
    # Get just the filename (e.g., "28_particles.csv")
    get_filename_component(FILE_NAME "${SOURCE_FILE}" NAME)

    # Define the destination path
    set(DEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/particle_locations/${FILE_NAME}")

    # Copy the file at configure time (similarly to configure_file COPYONLY)
    configure_file(
        ${SOURCE_FILE}
        ${DEST_FILE}
        COPYONLY
    )
endforeach()
