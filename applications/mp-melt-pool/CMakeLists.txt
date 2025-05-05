setup_application()

# Tests
file(GLOB CHILD_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/cases ${CMAKE_CURRENT_SOURCE_DIR}/cases/*)

foreach(subdir ${CHILD_DIRS})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cases/${subdir}/tests)
    add_subdirectory(cases/${subdir}/tests)
  endif()
endforeach()
