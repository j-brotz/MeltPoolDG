setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_COMP_FLOW
    APPLICATION_NAME "mp-compressible-flow"
    DESCRIPTION "Build compressible-flow executables"
)

if (APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(unit_tests)
endif()
