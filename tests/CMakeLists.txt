ADD_SUBDIRECTORY(tests_not_robust)

SET(TEST_LIBRARIES ${projectName})
SET(TEST_TARGET ${projectName})
DEAL_II_PICKUP_TESTS()

