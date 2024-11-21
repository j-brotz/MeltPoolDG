# Collect all .cpp files in the current directory
file(GLOB SOURCE_FILES "*.cpp")

# Check the number of matching files
list(LENGTH SOURCE_FILES NUM_FILES)

if(NUM_FILES EQUAL 1)
  # do nothing
elseif(NUM_FILES GREATER 1)
  message(FATAL_ERROR "Error: More than one file matches '*.cpp'. Found: ${SOURCE_FILES}")
else()
  message(FATAL_ERROR "Error: No file matches '*.cpp'.")
endif()

# Recursively collect all .cpp files in subdirectories
file(GLOB_RECURSE CASES "**/*.cpp")

# Recursively collect all .hpp files in subdirectories
file(GLOB_RECURSE CASES_HEADERS "**/*.hpp")

# Loop over each source file found in SOURCE_FILES
foreach (source_file ${SOURCE_FILES})
  # Extract the name of the executable from the current directory
  get_filename_component(exec ${CMAKE_CURRENT_SOURCE_DIR} NAME)

  # Notify about the executable being created
  message("-- Building executable: ${exec}")

  # Add an executable target with the current source file, all cases, and headers
  add_executable(${exec} ${source_file} ${CASES} ${CASES_HEADERS})
  
  DEAL_II_SETUP_TARGET(${exec})
  
  # Link additional libraries (libName is assumed to be defined elsewhere)
  target_link_libraries(${exec} ${libName})
endforeach (source_file ${SOURCE_FILES})

# Tests
ADD_SUBDIRECTORY(cases/tests)
