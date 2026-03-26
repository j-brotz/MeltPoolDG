setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_HEAT
    APPLICATION_NAME "mp-heat-transfer"
    DESCRIPTION "Build heat-transfer executables"
)

if(APP_ENABLED)
    # Unit tests
    add_subdirectory(cases/unit_tests)

    # Tests
    file(GLOB CHILD_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/cases ${CMAKE_CURRENT_SOURCE_DIR}/cases/*)

    foreach(subdir ${CHILD_DIRS})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cases/${subdir}/tests)
            add_subdirectory(cases/${subdir}/tests)
        endif()
    endforeach()

endif()
