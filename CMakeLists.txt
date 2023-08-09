message("_________________________________________________________________________________")
message("                                                                                 ")
message("                              MeltPoolDG                                         ")
message("                                                                                 ")
message("_________________________________________________________________________________")

# this is the standard deal.II search mechanism, including check for Trilinos
# and p4est
cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 20)

project(MeltPoolDG)

# avoid in-source compilation
set(DEFAULT_OUT_OF_SOURCE_FOLDER "build")
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(
    FATAL_ERROR
      "In-source builds not allowed. Change to ${DEFAULT_OUT_OF_SOURCE_FOLDER} folder and run then cmake")
endif()

# ##############################################################################
# set default build type
# ##############################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
  message("-- Setting to default CMAKE_BUILD_TYPE Debug")
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##############################################################################
# find deal.II
# ##############################################################################

find_package(deal.II 9.3 QUIET 
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
if(NOT ${deal.II_FOUND})
  message(
    FATAL_ERROR
      "\n"
      "*** Could not locate deal.II. ***\n\n"
      "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
      "or set an environment variable \"DEAL_II_DIR\" that contains this path.")
endif()

message("-- deal.II found: ${DEAL_II_DIR}")

# ##############################################################################
# find p4est
# ##############################################################################
if(NOT DEAL_II_WITH_P4EST)
  message(
    FATAL_ERROR
      "\nThis library requires a deal.II installation built with support for p4est but it appears to be missing!\n"
  )
endif()

# ##############################################################################
# find trilinos
# ##############################################################################
if(NOT DEAL_II_WITH_TRILINOS)
  message(
    FATAL_ERROR
      "\nThis library requires a deal.II installation built with support for Trilinos but it appears to be missing!\n"
  )
endif()
deal_ii_initialize_cached_variables()

# ##############################################################################
# build meltpooldg library
# ##############################################################################
set(libName "meltpooldg")

# query git information
DEAL_II_INITIALIZE_CACHED_VARIABLES()
DEAL_II_QUERY_GIT_INFORMATION(MELTPOOLDG)
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/meltpooldg/utilities/revision.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/meltpooldg/utilities/revision.hpp)

# Set the include directory and the name of the project
include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)

file(GLOB_RECURSE sources "${CMAKE_SOURCE_DIR}/source/**/*.cpp" "${CMAKE_SOURCE_DIR}/simulations/*.cpp")

add_library(${libName} ${sources})

TARGET_COMPILE_DEFINITIONS(${libName} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

deal_ii_setup_target(${libName})


if(${DEAL_II_VERSION_MAJOR} GREATER_EQUAL 10)
  message("-- Enabling early deprecation via #define PUBLIC DEAL_II_DEPRECATED_EARLY=[[deprecated]]" )
  add_compile_definitions(${libName} PUBLIC DEAL_II_DEPRECATED_EARLY=[[deprecated]])
endif()

# ##############################################################################
# create executable
# ##############################################################################

set(projectName "meltpooldg-temp")

message("-- building executable: meltpooldg")
add_executable(${projectName} meltpooldg.cpp)
deal_ii_setup_target(${projectName})

target_link_libraries(${projectName} ${libName})
# rename executable to meltpooldg
set_target_properties(${projectName} PROPERTIES OUTPUT_NAME ${libName})

# ##############################################################################
# link to adaflo (optional)
# ##############################################################################

find_library(ADAFLO adaflo HINTS ${ADAFLO_LIB})
if(ADAFLO)
  message("-- Adaflo found: ${ADAFLO}")
  target_compile_definitions(${projectName} PUBLIC MELT_POOL_DG_WITH_ADAFLO)
  target_include_directories(${projectName} PUBLIC ${ADAFLO_INCLUDE})
  target_link_libraries(${projectName} ${ADAFLO})
  target_compile_definitions(${libName} PUBLIC MELT_POOL_DG_WITH_ADAFLO)
  target_include_directories(${libName} PUBLIC ${ADAFLO_INCLUDE})
  target_link_libraries(${libName} ${ADAFLO})
  set(DEAL_II_WITH_ADAFLO_SUPPORT ON)
endif()

# Define custom targets to easily switch the build type:
add_custom_target(
  debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT
    "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
)

add_custom_target(
  release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT
    "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
)

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  set(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
endif()

# tests
set(_n_tests "0")
enable_testing()
add_subdirectory(tests)
add_subdirectory(unit_tests)
