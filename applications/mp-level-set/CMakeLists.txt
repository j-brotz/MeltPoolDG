setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_LS
    APPLICATION_NAME "mp-level-set"
    DESCRIPTION "Build level-set executables"
)

if (APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(unit_tests)
endif()
