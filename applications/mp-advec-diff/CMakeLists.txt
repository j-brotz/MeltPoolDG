setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_ADVEC_DIFF
    APPLICATION_NAME "mp-advec-diff"
    DESCRIPTION "Build advection-diffusion executables"
)

if (APP_ENABLED)
    add_subdirectory(cases/tests)
    add_subdirectory(cases/unit_tests)
endif()
