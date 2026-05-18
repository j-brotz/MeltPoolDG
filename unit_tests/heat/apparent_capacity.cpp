#include <gtest/gtest.h>

#include <deal.II/base/vectorization.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/heat/apparent_capacity.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

using namespace MeltPoolDG;
using namespace MeltPoolDG::Heat;
using namespace dealii;

namespace
{
  struct ApparentCapacityTestParameters
  {
    ApparentCapacityType type;
    bool                 expect_zero_at_bounds;
  };


  double
  midpoint_rule(const std::function<double(double)> &function,
                const double                         lower,
                const double                         upper,
                const unsigned int                   num_steps)
  {
    const double step_size = (upper - lower) / num_steps;

    double value = 0.0;

    for (unsigned int i = 0; i < num_steps; ++i)
      {
        const double x = lower + (i + 0.5) * step_size;
        value += function(x);
      }

    return value * step_size;
  }


  double
  calculate_solid_fraction(const double temperature, const MaterialData<double> &material)
  {
    const double inv_mushy_interval =
      1.0 / (material.liquidus_temperature - material.solidus_temperature);

    const double value = (material.liquidus_temperature - temperature) * inv_mushy_interval;

    return std::clamp(value, 0.0, 1.0);
  }



  MaterialData<double>
  create_material_data(const ApparentCapacityType apparent_capacity_type)
  {
    MaterialData<double> material;

    material.latent_heat_of_fusion = 338e3;
    material.solidus_temperature   = 271.0;
    material.liquidus_temperature  = 275.0;

    material.apparent_capacity_type = apparent_capacity_type;

    return material;
  }
} // namespace



class ApparentCapacityTest : public testing::TestWithParam<ApparentCapacityTestParameters>
{};



TEST_P(ApparentCapacityTest, ConstraintsAtSolidusAndLiquidusTemperature)
{
  const double tol = 1e-6;

  const auto parameters = GetParam();

  if (!parameters.expect_zero_at_bounds)
    GTEST_SKIP() << "This apparent-capacity model is not expected to vanish at "
                    "the mushy-zone bounds.";

  const auto material = create_material_data(parameters.type);

  ApparentCapacity<double> apparent_capacity(material);

  const auto evaluate = [&](const double temperature) {
    const VectorizedArray<double> solid_fraction = calculate_solid_fraction(temperature, material);

    return apparent_capacity.evaluate(solid_fraction)[0];
  };

  const auto evaluate_derivative = [&](const double temperature) {
    const VectorizedArray<double> solid_fraction = calculate_solid_fraction(temperature, material);

    return apparent_capacity.compute_temperature_derivative(solid_fraction)[0];
  };

  EXPECT_NEAR(evaluate(material.solidus_temperature), 0.0, tol);
  EXPECT_NEAR(evaluate_derivative(material.solidus_temperature), 0.0, tol);

  EXPECT_NEAR(evaluate(material.liquidus_temperature), 0.0, tol);
  EXPECT_NEAR(evaluate_derivative(material.liquidus_temperature), 0.0, tol);
}



TEST_P(ApparentCapacityTest, IntegralEqualsLatentHeat)
{
  const double tol = 1e-8;

  const auto material = create_material_data(GetParam().type);

  ApparentCapacity<double> apparent_capacity(material);

  const auto evaluate = [&](const double temperature) {
    const VectorizedArray<double> solid_fraction = calculate_solid_fraction(temperature, material);

    return apparent_capacity.evaluate(solid_fraction)[0];
  };

  const double integral =
    midpoint_rule(evaluate, material.solidus_temperature, material.liquidus_temperature, 1000);

  const double relative_error =
    std::abs((integral - material.latent_heat_of_fusion) / material.latent_heat_of_fusion);

  EXPECT_NEAR(relative_error, 0.0, tol)
    << "Got integral = " << integral
    << ", expected latent heat = " << material.latent_heat_of_fusion;
}



TEST_P(ApparentCapacityTest, TemperatureDerivativeMatchesFiniteDifferences)
{
  const double tol = 1e-5;
  const double eps = 1e-5;

  const auto material = create_material_data(GetParam().type);

  ApparentCapacity<double> apparent_capacity(material);

  const auto evaluate = [&](const double temperature) {
    const VectorizedArray<double> solid_fraction = calculate_solid_fraction(temperature, material);

    return apparent_capacity.evaluate(solid_fraction)[0];
  };

  const auto evaluate_derivative = [&](const double temperature) {
    const VectorizedArray<double> solid_fraction = calculate_solid_fraction(temperature, material);

    return apparent_capacity.compute_temperature_derivative(solid_fraction)[0];
  };

  double       temperature = material.solidus_temperature + 0.1;
  const double step        = (material.liquidus_temperature - material.solidus_temperature) / 5.0;

  for (unsigned int i = 0; i <= 2; ++i)
    {
      const double expected =
        (evaluate(temperature + eps) - evaluate(temperature - eps)) / (2.0 * eps);

      const double value = evaluate_derivative(temperature);

      EXPECT_NEAR(value, expected, tol * std::max(1.0, std::abs(expected)))
        << "T = " << temperature;

      temperature += step;
    }
}



INSTANTIATE_TEST_SUITE_P(
  AllModels,
  ApparentCapacityTest,
  testing::Values(ApparentCapacityTestParameters{ApparentCapacityType::poly4_bell, true},
                  ApparentCapacityTestParameters{ApparentCapacityType::constant, false},
                  ApparentCapacityTestParameters{ApparentCapacityType::qlq, false}));
