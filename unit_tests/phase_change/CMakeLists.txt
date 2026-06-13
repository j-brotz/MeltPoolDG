set(TEST_LIBRARIES ${meltpooldg_lib})
set(TEST_TARGET ${meltpooldg_lib})
deal_ii_pickup_tests()

mpdg_setup_serial_unit_tests(
    TEST_SOURCE_FILES
    evaporation_model_knight.cpp
)
