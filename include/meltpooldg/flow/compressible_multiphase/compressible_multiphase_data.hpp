/**
 * @brief Data structure, which contains parameters specifically for the phase coupling of compressible multiphase
 * simulations.
 */

#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/better_enum.hpp>

namespace MeltPoolDG::Multiphase
{
  BETTER_ENUM(InterfaceNumericalMethod, char, HLLP0_and_Nitsche, penalty)

  template <typename number = double>
  struct CompressibleMultiphaseData
  {
    // evaporation mass flux
    // TODO: use Hertz-Knudsen theory and enable constant evaporation mass flux for testing
    double m_dot_evap = 0.;

    // numerical method for interface jump enforcement
    InterfaceNumericalMethod interface_numerical_method = InterfaceNumericalMethod::penalty;

    // density constraint penalty factor
    double density_constraint_penalty_factor = 10.;

    // temperature constraint penalty factor
    double temperature_constraint_penalty_factor = 10.;

    // TODO: remove from input file, only temporary required for testing
    // target values for density and temperature for gas phase, required for penalty approach of
    // interface jump conditions
    double target_density_gas_phase     = 0.;
    double target_temperature_gas_phase = 0.;

    // symmetric interior penalty parameter for the viscous interface term
    double symm_int_penalty_parameter_interface = 1.;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("multiphase");
      {
        prm.add_parameter("evaporation mass flux", m_dot_evap, "Evaporation mass flux.");
        prm.add_parameter("interface numerical method",
                          interface_numerical_method,
                          "Numerical method for enforcing interface jump conditions.");
        prm.add_parameter("density constraint penalty factor",
                          density_constraint_penalty_factor,
                          "Density constraint penalty factor.");
        prm.add_parameter("temperature constraint penalty factor",
                          temperature_constraint_penalty_factor,
                          "Temperature constraint penalty factor.");
        prm.add_parameter("symm int penalty parameter interface",
                          symm_int_penalty_parameter_interface,
                          "Symmetric interior penalty parameter for interface term.");
        prm.add_parameter("target density gas phase",
                          target_density_gas_phase,
                          "Target density of gas phase.");
        prm.add_parameter("target temperature gas phase",
                          target_temperature_gas_phase,
                          "Target temperature of gas phase.");
      }
      prm.leave_subsection();
    };
  };
} // namespace MeltPoolDG::Multiphase