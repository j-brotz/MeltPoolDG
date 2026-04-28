setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_MELTPOOL_VAPOR_FLOW
    APPLICATION_NAME "mp-meltpool-vapor-flow"
    DESCRIPTION "Build meltpool-vapor-flow executables"
)

if (APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(unit_tests)
endif()
