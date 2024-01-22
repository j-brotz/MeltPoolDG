#include <gtest/gtest.h>

#include <meltpooldg/core/material_data.hpp>

#include "test_utils/utils.hpp"

namespace MeltPoolDG
{
  /**
   * Test suite for MaterialData::create_Ti64_material_data()
   *
   * This test verifies that the Ti-6Al-4V (Titanium alloy) material data is created with
   * the correct properties as documented.
   */
  class Ti64MaterialDataTest : public ::testing::Test
  {
  protected:
    Ti64MaterialDataTest() = default;

    void
    SetUp() override
    {
      // Create Ti64 material data
      material_data = MeltPoolDG::MaterialData<double>::create_Ti64_material_data();
    }

    MeltPoolDG::MaterialData<double> material_data;
  };


  /**
   * Test gas phase properties for Ti-6Al-4V
   */
  TEST_F(Ti64MaterialDataTest, GasPhaseProperties)
  {
    EXPECT_DOUBLE_EQ(material_data.gas.thermal_conductivity, 0.02863); // W / (m K)
    EXPECT_DOUBLE_EQ(material_data.gas.specific_heat_capacity, 11.3);  // J / (kg K)
    EXPECT_DOUBLE_EQ(material_data.gas.density, 44.1);                 // kg / m³
    EXPECT_DOUBLE_EQ(material_data.gas.dynamic_viscosity, 0.00035);    // kg / (m s)
  }

  /**
   * Test liquid phase properties for Ti-6Al-4V
   */
  TEST_F(Ti64MaterialDataTest, LiquidPhaseProperties)
  {
    EXPECT_DOUBLE_EQ(material_data.liquid.thermal_conductivity, 28.63);    // W / (m K)
    EXPECT_DOUBLE_EQ(material_data.liquid.specific_heat_capacity, 1130.0); // J / (kg K)
    EXPECT_DOUBLE_EQ(material_data.liquid.density, 4087.0);                // kg / m³
    EXPECT_DOUBLE_EQ(material_data.liquid.dynamic_viscosity, 0.0035);      // kg / (m s)
  }

  /**
   * Test solid phase properties for Ti-6Al-4V
   */
  TEST_F(Ti64MaterialDataTest, SolidPhaseProperties)
  {
    EXPECT_DOUBLE_EQ(material_data.solid.thermal_conductivity, 28.63);    // W / (m K)
    EXPECT_DOUBLE_EQ(material_data.solid.specific_heat_capacity, 1130.0); // J / (kg K)
    EXPECT_DOUBLE_EQ(material_data.solid.density, 4087.0);                // kg / m³
    EXPECT_DOUBLE_EQ(material_data.solid.dynamic_viscosity, 0.35);        // kg / (m s)
  }

  /**
   * Test that liquid and solid share the same thermal properties
   */
  TEST_F(Ti64MaterialDataTest, LiquidSolidThermalPropertiesAreEqual)
  {
    EXPECT_DOUBLE_EQ(material_data.liquid.thermal_conductivity,
                     material_data.solid.thermal_conductivity);
    EXPECT_DOUBLE_EQ(material_data.liquid.specific_heat_capacity,
                     material_data.solid.specific_heat_capacity);
    EXPECT_DOUBLE_EQ(material_data.liquid.density, material_data.solid.density);
  }

  /**
   * Test temperature thresholds for Ti-6Al-4V
   */
  TEST_F(Ti64MaterialDataTest, TemperatureThresholds)
  {
    EXPECT_DOUBLE_EQ(material_data.solidus_temperature, 1933.0);  // K
    EXPECT_DOUBLE_EQ(material_data.liquidus_temperature, 2200.0); // K
    EXPECT_DOUBLE_EQ(material_data.boiling_temperature, 3133.0);  // K
  }

  /**
   * Test that liquidus temperature is greater than solidus temperature
   */
  TEST_F(Ti64MaterialDataTest, LiquidusSolidusTemperatureOrder)
  {
    EXPECT_GT(material_data.liquidus_temperature, material_data.solidus_temperature);
  }

  /**
   * Test that boiling temperature is greater than liquidus temperature
   */
  TEST_F(Ti64MaterialDataTest, BoilingTemperatureGreaterThanLiquidus)
  {
    EXPECT_GT(material_data.boiling_temperature, material_data.liquidus_temperature);
  }

  /**
   * Test thermodynamic properties for Ti-6Al-4V
   */
  TEST_F(Ti64MaterialDataTest, ThermodynamicProperties)
  {
    EXPECT_DOUBLE_EQ(material_data.latent_heat_of_evaporation, 8.84e6);             // J / kg
    EXPECT_DOUBLE_EQ(material_data.molar_mass, 4.78e-2);                            // kg / mol
    EXPECT_DOUBLE_EQ(material_data.specific_enthalpy_reference_temperature, 538.0); // K
  }

  /**
   * Test that all material properties are positive (physical validity check)
   */
  TEST_F(Ti64MaterialDataTest, AllPropertiesArePositive)
  {
    // Gas properties
    EXPECT_GT(material_data.gas.thermal_conductivity, 0.0);
    EXPECT_GT(material_data.gas.specific_heat_capacity, 0.0);
    EXPECT_GT(material_data.gas.density, 0.0);
    EXPECT_GT(material_data.gas.dynamic_viscosity, 0.0);

    // Liquid properties
    EXPECT_GT(material_data.liquid.thermal_conductivity, 0.0);
    EXPECT_GT(material_data.liquid.specific_heat_capacity, 0.0);
    EXPECT_GT(material_data.liquid.density, 0.0);
    EXPECT_GT(material_data.liquid.dynamic_viscosity, 0.0);

    // Solid properties
    EXPECT_GT(material_data.solid.thermal_conductivity, 0.0);
    EXPECT_GT(material_data.solid.specific_heat_capacity, 0.0);
    EXPECT_GT(material_data.solid.density, 0.0);
    EXPECT_GT(material_data.solid.dynamic_viscosity, 0.0);

    // Thermodynamic properties
    EXPECT_GT(material_data.latent_heat_of_evaporation, 0.0);
    EXPECT_GT(material_data.molar_mass, 0.0);
  }

  /**
   * Test consistency between gas and liquid thermal properties
   *
   * Liquid thermal conductivity should generally be higher than gas
   */
  TEST_F(Ti64MaterialDataTest, GasLiquidThermalConductivityComparison)
  {
    EXPECT_GT(material_data.liquid.thermal_conductivity, material_data.gas.thermal_conductivity);
  }

  /**
   * Test consistency between gas and liquid density
   *
   * Liquid density should generally be much higher than gas density
   */
  TEST_F(Ti64MaterialDataTest, GasLiquidDensityComparison)
  {
    EXPECT_GT(material_data.liquid.density, material_data.gas.density);
  }

  /**
   * Test that the function works with different template types
   */
  TEST(Ti64MaterialDataTemplateTest, FloatTemplateInstantiation)
  {
    auto material_data_float = MeltPoolDG::MaterialData<float>::create_Ti64_material_data();
    EXPECT_FLOAT_EQ(static_cast<float>(material_data_float.solidus_temperature), 1933.0f);
  }

  /**
   * Test that material data passes heat transfer parameter checks
   */
  TEST_F(Ti64MaterialDataTest, PassesHeatTransferParameterChecks)
  {
    // Should not throw for two-phase flow with heat transfer
    EXPECT_NO_THROW(material_data.check_parameters_heat_transfer(true, false));

    // Should not throw for solidification
    EXPECT_NO_THROW(material_data.check_parameters_heat_transfer(false, true));

    // Should not throw for both two-phase and solidification
    EXPECT_NO_THROW(material_data.check_parameters_heat_transfer(true, true));
  }
} // namespace MeltPoolDG
