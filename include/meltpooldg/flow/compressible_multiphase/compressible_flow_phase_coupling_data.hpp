/**
 * @brief Data structure, which contains parameters specifically for the phase coupling of
 * compressible multiphase simulations.
 */

#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>

namespace MeltPoolDG::Multiphase
{
  BETTER_ENUM(InterfaceNumericalMethod, char, HLLP0_and_SIPG, HLLP0_and_penalty, penalty)

  template <typename number>
  struct CompressibleFlowPhaseCouplingData
  {
    // evaporation mass flux
    // TODO: use Hertz-Knudsen theory and enable constant evaporation mass flux for testing
    number m_dot_evap = 0.;

    // energy flux jump (q_liquid - q_gas) (sum of evaporative enthalpy loss and laser heat source)
    // TODO: the parameter is only temporarily relevant
    number delta_q = 0.;

    // numerical method for interface jump enforcement
    InterfaceNumericalMethod type = InterfaceNumericalMethod::HLLP0_and_SIPG;

    struct Penalty
    {
      struct Coefficients
      {
        // density constraint penalty factor
        number density = std::numeric_limits<number>::max();

        // temperature constraint penalty factor
        number temperature = std::numeric_limits<number>::max();
      } coefficients;

      struct TargetValues
      {
        // TODO: remove from input file, only temporary required for testing
        // target values for density and temperature for gas phase, required for penalty approach of
        // interface jump conditions
        number density_gas_phase     = 0.;
        number temperature_gas_phase = 0.;
      } target_values;

    } penalty;

    struct HLLP0_SIPG
    {
      // symmetric interior penalty parameter for the viscous interface term
      number interior_penalty_parameter_interface = 1.;
      // temperature jump (T_liquid - T_gas)
      number delta_T = 0.;
    } hllp0_and_sipg;

    struct HLLP0_penalty
    {
      // penalty parameter for temperature jump constraint
      number penalty_parameter_temperature_jump = 1.;
      // temperature jump (T_liquid - T_gas)
      number delta_T = 0.;
    } hllp0_and_penalty;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("compressible flow phase coupling");
      {
        // TODO: add parameter "bool const_m_dot_evap" when Hertz-Knudsen theory is enabled
        prm.add_parameter(
          "evaporation mass flux",
          m_dot_evap,
          "Evaporation mass"
          " flux at the phase interface. This parameter is only relevant if a given "
          "mass flux is considered, i.e. no thermodynamical model (Hertz-Knudsen "
          "theory) is applied.",
          dealii::Patterns::Double());
        prm.add_parameter("delta q",
                          delta_q,
                          "Delta q (sum of evaporative enthalpy loss and laser heat source).",
                          dealii::Patterns::Double());
        prm.add_parameter("type",
                          type,
                          "Numerical method for enforcing interface jump conditions. "
                          "The options are \"penalty\", \"HLLP0_and_SIPG\", \"HLLP0_and_penalty\".",
                          dealii::Patterns::Selection("penalty|HLLP0_and_SIPG|HLLP0_and_penalty"));
        prm.enter_subsection("penalty");
        prm.enter_subsection("coefficients");
        prm.add_parameter(
          "density",
          penalty.coefficients.density,
          "Density constraint penalty factor for enforcing the density in the gas phase. "
          "The minimum value is 0.",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter(
          "temperature",
          penalty.coefficients.temperature,
          "Temperature constraint penalty factor for enforcing the temperature in the "
          "gas phase. The minimum value is 0.",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.leave_subsection();
        // TODO: In the case that the Hertz-Knudsen theory is considered, no target values should be
        // set in the user input
        prm.enter_subsection("target values");
        prm.add_parameter("density gas phase",
                          penalty.target_values.density_gas_phase,
                          "Target density of gas phase.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("temperature gas phase",
                          penalty.target_values.temperature_gas_phase,
                          "Target temperature of gas phase.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.leave_subsection();
        prm.leave_subsection();
        prm.enter_subsection("HLLP0 and SIPG");
        prm.add_parameter(
          "interior penalty parameter interface",
          hllp0_and_sipg.interior_penalty_parameter_interface,
          "Symmetric interior penalty parameter for the interface term. A good first "
          "guess is O(1/cell_size). Increase the parameter in the case of instabilities at the "
          "interface. The minimum value is 0.",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("delta T",
                          hllp0_and_sipg.delta_T,
                          "Temperature jump at the interface.",
                          dealii::Patterns::Double());
        prm.leave_subsection();
        prm.enter_subsection("HLLP0 and penalty");
        prm.add_parameter("penalty parameter temperature jump",
                          hllp0_and_penalty.penalty_parameter_temperature_jump,
                          "Penalty parameter for the temperature jump constraint.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("delta T",
                          hllp0_and_penalty.delta_T,
                          "Temperature jump at the interface.",
                          dealii::Patterns::Double());
        prm.leave_subsection();
      }
      prm.leave_subsection();
    };

    void
    post()
    {
      if (type == InterfaceNumericalMethod::penalty)
        {
          AssertThrow(penalty.coefficients.density != std::numeric_limits<number>::max(),
                      dealii::ExcMessage(
                        "You have to set the density constraint penalty factor for penalty "
                        "method."));
          AssertThrow(penalty.coefficients.temperature != std::numeric_limits<number>::max(),
                      dealii::ExcMessage(
                        "You have to set the temperature constraint penalty factor for penalty "
                        "method."));
        }
    };
  };
} // namespace MeltPoolDG::Multiphase
