setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_RTE
    APPLICATION_NAME "mp-radiative-transport"
    DESCRIPTION "Build radiative-transport executables"
)

if (APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(unit_tests)
endif()
