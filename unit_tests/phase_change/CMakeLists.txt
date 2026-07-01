set(TEST_LIBRARIES ${MELTPOOLDG_LIB})
set(TEST_TARGET ${MELTPOOLDG_LIB})
deal_ii_pickup_tests()

mpdg_setup_serial_unit_tests(
    TEST_SOURCE_FILES
    evaporation_model_knight.cpp
)
