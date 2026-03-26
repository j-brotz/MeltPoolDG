setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_CFD_DEM
    APPLICATION_NAME "mp-cfd-dem"
    DESCRIPTION "Build CFD-DEM executables"
)

if (APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(cases/unit_tests)
endif()