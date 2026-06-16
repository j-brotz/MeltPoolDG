#include <gtest/gtest.h>

#include <deal.II/base/vectorization.h>

#include <meltpooldg/phase_change/evaporation_model_knight.hpp>

#include <algorithm>
#include <array>
#include <cmath>

using namespace MeltPoolDG;

namespace
{
  struct TestCase
  {
    double Ma_gas;
    double T_liquid;
    double expected_m_dot;
    double expected_T_jump;
  };

  constexpr double atmospheric_pressure                        = 1.e5;
  constexpr double boiling_temperature_at_atmospheric_pressure = 3133.;
  constexpr double latent_heat_of_evaporation                  = 8.84e6;
  constexpr double specific_gas_constant                       = 173.93;
  constexpr double specific_heat_ratio_vapor                   = 5. / 3.;

  constexpr std::array<TestCase, 9> reference_cases = {
    {TestCase{.Ma_gas = 0.1, .T_liquid = 3100., .expected_m_dot = 0., .expected_T_jump = 0.},
     TestCase{.Ma_gas          = 0.1,
              .T_liquid        = 3300.,
              .expected_m_dot  = 32.683938321679236,
              .expected_T_jump = 130.81427466446257},
     TestCase{.Ma_gas          = 0.1,
              .T_liquid        = 3500.,
              .expected_m_dot  = 76.519902933140543,
              .expected_T_jump = 138.74241252291495},
     TestCase{.Ma_gas = 0.5, .T_liquid = 3100., .expected_m_dot = 0., .expected_T_jump = 0.},
     TestCase{.Ma_gas          = 0.5,
              .T_liquid        = 3300.,
              .expected_m_dot  = 89.501510330964891,
              .expected_T_jump = 603.3428412049916},
     TestCase{.Ma_gas          = 0.5,
              .T_liquid        = 3500.,
              .expected_m_dot  = 209.54166586320497,
              .expected_T_jump = 639.90907400529431},
     TestCase{.Ma_gas = 1.0, .T_liquid = 3100., .expected_m_dot = 0., .expected_T_jump = 0.},
     TestCase{.Ma_gas          = 1.0,
              .T_liquid        = 3300.,
              .expected_m_dot  = 97.616375809273194,
              .expected_T_jump = 1091.9157127994849},
     TestCase{.Ma_gas          = 1.0,
              .T_liquid        = 3500.,
              .expected_m_dot  = 228.54025509698064,
              .expected_T_jump = 1158.0924226661205}}};

  constexpr double relative_tolerance = 1.e-12;

  TEST(EvaporationModelKnight, ScalarReference)
  {
    Evaporation::EvaporationModelKnight<double> model(atmospheric_pressure,
                                                      boiling_temperature_at_atmospheric_pressure,
                                                      latent_heat_of_evaporation,
                                                      specific_gas_constant,
                                                      specific_heat_ratio_vapor);

    for (const auto &test_case : reference_cases)
      {
        SCOPED_TRACE("Ma_gas=" + std::to_string(test_case.Ma_gas) +
                     ", T_liquid=" + std::to_string(test_case.T_liquid));

        model.reinit(test_case.T_liquid, test_case.Ma_gas);

        const double m_dot  = model.get_evaporative_mass_flux();
        const double T_jump = model.get_temperature_jump();

        EXPECT_NEAR(m_dot,
                    test_case.expected_m_dot,
                    relative_tolerance * std::max(1., std::abs(test_case.expected_m_dot)));

        EXPECT_NEAR(T_jump,
                    test_case.expected_T_jump,
                    relative_tolerance * std::max(1., std::abs(test_case.expected_T_jump)));
      }
  }

  TEST(EvaporationModelKnight, VectorizedReference)
  {
    constexpr unsigned int n_lanes = dealii::VectorizedArray<double>::size();

    Evaporation::EvaporationModelKnight<double, dealii::VectorizedArray<double>> model(
      atmospheric_pressure,
      boiling_temperature_at_atmospheric_pressure,
      latent_heat_of_evaporation,
      specific_gas_constant,
      specific_heat_ratio_vapor);

    for (unsigned int first = 0; first < reference_cases.size(); first += n_lanes)
      {
        // Determine the active entries in the vectorized array
        const unsigned int n_active_entries =
          std::min<unsigned int>(n_lanes, reference_cases.size() - first);

        dealii::VectorizedArray<double> T_liquid;
        dealii::VectorizedArray<double> Ma_gas;

        std::array<TestCase, n_lanes> lane_cases;

        for (unsigned int lane = 0; lane < n_lanes; ++lane)
          {
            const unsigned int idx =
              std::min<unsigned int>(first + lane, reference_cases.size() - 1);

            lane_cases[lane] = reference_cases[idx];

            T_liquid[lane] = lane_cases[lane].T_liquid;
            Ma_gas[lane]   = lane_cases[lane].Ma_gas;
          }

        model.reinit(T_liquid, Ma_gas, n_active_entries);

        const dealii::VectorizedArray<double> m_dot  = model.get_evaporative_mass_flux();
        const dealii::VectorizedArray<double> T_jump = model.get_temperature_jump();

        for (unsigned int lane = 0; lane < n_active_entries; ++lane)
          {
            SCOPED_TRACE("packet_start=" + std::to_string(first) +
                         ", lane=" + std::to_string(lane) +
                         ", Ma_gas=" + std::to_string(lane_cases[lane].Ma_gas) +
                         ", T_liquid=" + std::to_string(lane_cases[lane].T_liquid));

            EXPECT_NEAR(m_dot[lane],
                        lane_cases[lane].expected_m_dot,
                        relative_tolerance *
                          std::max(1., std::abs(lane_cases[lane].expected_m_dot)));

            EXPECT_NEAR(T_jump[lane],
                        lane_cases[lane].expected_T_jump,
                        relative_tolerance *
                          std::max(1., std::abs(lane_cases[lane].expected_T_jump)));
          }
      }
  }
} // namespace
