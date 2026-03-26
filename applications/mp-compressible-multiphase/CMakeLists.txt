setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_COMP_MULTIPHASE
    APPLICATION_NAME "mp-compressible-multiphase"
    DESCRIPTION "Build compressible-multiphase executables"
)

if (APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(cases/unit_tests)
endif()