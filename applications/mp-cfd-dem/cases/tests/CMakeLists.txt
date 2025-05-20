set(TEST_TARGET "mp-cfd-dem")
deal_ii_pickup_tests("mp-cfd-dem")

file(GLOB CSV_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.csv")
foreach (CSV_FILE ${CSV_FILES})
    get_filename_component(FILENAME ${CSV_FILE} NAME)
    configure_file(${CSV_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach ()
