SET(TEST_LIBRARIES ${libName})
SET(TEST_TARGET ${libName})
DEAL_II_PICKUP_TESTS()


SET(_n_tests "0")
FILE(GLOB _tests ${_testexe}*.output)
LIST (SORT _tests)
FOREACH(_test ${_tests})
  MATH(EXPR _n_tests "${_n_tests} + 1")
ENDFOREACH()
MESSAGE(STATUS "Added ${_n_tests} unit tests")

