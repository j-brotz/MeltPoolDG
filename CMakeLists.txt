message("_________________________________________________________________________________")
message("                                                                                 ")
message("                              MeltPoolDG                                         ")
message("                                                                                 ")
message("_________________________________________________________________________________")

# this is the standard deal.II search mechanism, including check for Trilinos
# and p4est
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

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

find_package(deal.II 9.6 QUIET
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
# find adaflo
# ##############################################################################
find_library(ADAFLO adaflo HINTS ${ADAFLO_LIB})
if(ADAFLO)
    add_compile_definitions(ADAFLO_SUPPORT)
    set(DEAL_II_WITH_ADAFLO_SUPPORT ON)
    set(MELT_POOL_DG_WITH_ADAFLO TRUE)
    message(STATUS "Adaflo found: ${ADAFLO}")
else()
    set(MELT_POOL_DG_WITH_ADAFLO FALSE)
    message(STATUS "Adaflo not found. Setting up project without adaflo support.")
endif()

# ##############################################################################
# build meltpooldg library
# ##############################################################################

# query git information
DEAL_II_INITIALIZE_CACHED_VARIABLES()
DEAL_II_QUERY_GIT_INFORMATION(MELTPOOLDG)
CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meltpooldg/utilities/revision.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/meltpooldg/utilities/revision.hpp)

# Set the include directory and the name of the project
include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)

file(GLOB_RECURSE sources
    "${CMAKE_SOURCE_DIR}/source/**/*.cpp"
    "${CMAKE_SOURCE_DIR}/simulations/*.cpp")


set(meltpooldg_lib "meltpooldg")

add_library(${meltpooldg_lib} ${sources})

if(ADAFLO)
    target_compile_definitions(${meltpooldg_lib} PUBLIC MELT_POOL_DG_WITH_ADAFLO)
    target_include_directories(${meltpooldg_lib} PUBLIC ${ADAFLO_INCLUDE})
    target_link_libraries(${meltpooldg_lib} ${ADAFLO})
endif()

TARGET_COMPILE_DEFINITIONS(${meltpooldg_lib} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

deal_ii_setup_target(${meltpooldg_lib})

if(${DEAL_II_VERSION_MAJOR} GREATER_EQUAL 10)
    message("-- Enabling early deprecation via #define PUBLIC DEAL_II_DEPRECATED_EARLY=[[deprecated]]")
    add_compile_definitions(${meltpooldg_lib} PUBLIC DEAL_II_DEPRECATED_EARLY=[[deprecated]])
endif()

# ##############################################################################
# benchmarks
# ##############################################################################
option(MPDG_EANBLE_BENCHMARKING "Configure benchmarks" ON)
if(MPDG_EANBLE_BENCHMARKING)
    message(STATUS "Benchmarks: enabled")
    # Deactivate internal google benchmark tests
    set(BENCHMARK_ENABLE_TESTING OFF)
    # Ensure that google benchmark debug assertions are disabled
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")
    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.3)

    FetchContent_MakeAvailable(google_benchmark)
    include_directories(${google_benchmark_SOURCE_DIR}/src)
    add_subdirectory(benchmarks)
else()
    message(STATUS "Benchmarks: disabled")
endif()

# ##############################################################################
# unit tests of the library
# ##############################################################################
option(MPDG_ENABLE_TESTING "Configure tests" ON)

if(MPDG_ENABLE_TESTING)
    message(STATUS "Tests: enabled")

    fetchcontent_declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )
    fetchcontent_makeavailable(googletest)
    include(GoogleTest)

    set(DEAL_II_HAVE_TESTS_DIRECTORY ON)
    set(_n_tests "0")
    enable_testing()
    add_subdirectory(unit_tests)
else()
    message(STATUS "Tests: disabled")
endif()


# ##############################################################################
# Applications
# ##############################################################################
file(GLOB custom_application_cmake_commands "${CMAKE_SOURCE_DIR}/applications/cmake/*.cmake")
foreach(command ${custom_application_cmake_commands})
    include(${command})
endforeach()

option(MPDG_ENABLE_ADVEC_DIFF "Build advection-diffusion executables" ON)
option(MPDG_ENABLE_COMP_FLOW "Build compressible-flow executables" ON)
option(MPDG_ENABLE_COMP_MULTIPHASE "Build compressible-multiphase executables" ON)
option(MPDG_ENABLE_CFD_DEM "Build cfd-dem executables" ON)
option(MPDG_ENABLE_DEM "Build dem executables" ON)
option(MPDG_ENABLE_HEAT "Build heat-transfer executables" ON)
option(MPDG_ENABLE_RTE "Build radiative-transport executables" ON)
option(MPDG_ENABLE_REINIT "Build reinitialization executables" ON)
option(MPDG_ENABLE_LS "Build level-set executables" ON)
option(MPDG_ENABLE_MP "Build melt-pool executables" ON)

if(MPDG_ENABLE_ADVEC_DIFF)
    message(STATUS "Application: advection-diffusion enabled")
    add_subdirectory(applications/mp-advec-diff)
else()
    message(STATUS "Application: advection-diffusion disabled")
endif()

if(MPDG_ENABLE_COMP_FLOW)
    message(STATUS "Application: compressible-flow enabled")
    add_subdirectory(applications/mp-compressible-flow)
else()
    message(STATUS "Application: compressible-flow disabled")
endif()

if(MPDG_ENABLE_COMP_MULTIPHASE)
    message(STATUS "Application: compressible-multiphase enabled")
    add_subdirectory(applications/mp-compressible-multiphase)
else()
    message(STATUS "Application: compressible-multiphase disabled")
endif()

if(MPDG_ENABLE_CFD_DEM)
    message(STATUS "Application: cfd-dem enabled")
    add_subdirectory(applications/mp-cfd-dem)
else()
    message(STATUS "Application: cfd-dem disabled")
endif()

if(MPDG_ENABLE_DEM)
    message(STATUS "Application: dem enabled")
    add_subdirectory(applications/mp-dem)
else()
    message(STATUS "Application: dem disabled")
endif()

if(MPDG_ENABLE_HEAT)
    message(STATUS "Application: heat-transfer enabled")
    add_subdirectory(applications/mp-heat-transfer)
else()
    message(STATUS "Application: heat-transfer disabled")
endif()

if(MPDG_ENABLE_RTE)
    message(STATUS "Application: radiative-transport enabled")
    add_subdirectory(applications/mp-radiative-transport)
    set(MELT_POOL_DG_WITH_RTE TRUE)
else()
    message(STATUS "Application: radiative-transport disabled")
    set(MELT_POOL_DG_WITH_RTE FALSE)
endif()
target_compile_definitions(${meltpooldg_lib} PUBLIC MELT_POOL_DG_WITH_RTE)

if(MPDG_ENABLE_REINIT)
    message(STATUS "Application: reinitialization enabled")
    add_subdirectory(applications/mp-reinit)
else()
    message(STATUS "Application: reinitialization disabled")
endif()

if(MPDG_ENABLE_LS)
    message(STATUS "Application: level-set enabled")
    add_subdirectory(applications/mp-level-set)
else()
    message(STATUS "Application: level-set disabled")
endif()

if(MPDG_ENABLE_MP)
    message(STATUS "Application: melt-pool enabled")
    add_subdirectory(applications/mp-melt-pool)
else()
    message(STATUS "Application: melt-pool disabled")
endif()
