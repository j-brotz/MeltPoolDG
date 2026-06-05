#include <gtest/gtest.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/heat/laser_data.hpp>

using namespace MeltPoolDG;
using namespace MeltPoolDG::Heat;

TEST(LaserData, ScalesAbsorptivityIntoPowerForCutOperator)
{
  LaserData<double> laser;
  laser.power               = 100.0;
  laser.absorptivity_gas    = 0.4;
  laser.absorptivity_liquid = 0.4;

  MaterialData<double> material;

  laser.post(/*dim*/ 3,
             /*heat_use_volume_specific_thermal_capacity_for_phase_interpolation*/ false,
             material,
             /*heat_is_cut_operator*/ true);

  EXPECT_DOUBLE_EQ(laser.power, 40.0);
  EXPECT_DOUBLE_EQ(laser.absorptivity_gas, 1.0);
  EXPECT_DOUBLE_EQ(laser.absorptivity_liquid, 1.0);
}
