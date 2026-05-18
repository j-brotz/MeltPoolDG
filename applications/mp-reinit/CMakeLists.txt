setup_application(
    APP_ENABLED
    CONFIGURE_OPTION_NAME MPDG_ENABLE_REINIT
    APPLICATION_NAME "mp-reinit"
    DESCRIPTION "Build reinitialization executables"
)

if(APP_ENABLED)
    # Unit tests
    add_subdirectory(unit_tests)

    # Add tests in subdirectories
    file(GLOB CHILD_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/cases ${CMAKE_CURRENT_SOURCE_DIR}/cases/*)

    foreach(subdir ${CHILD_DIRS})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cases/${subdir}/tests)
            add_subdirectory(cases/${subdir}/tests)
        endif()
    endforeach()
endif()
