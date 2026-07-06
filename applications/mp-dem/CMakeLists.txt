setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_DEM
    APPLICATION_NAME "mp-dem"
    DESCRIPTION "Build DEM executables"
)

if(APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(unit_tests)
endif()