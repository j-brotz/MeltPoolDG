#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/utilities/better_enum.hpp>

namespace MeltPoolDG::Flow
{
  // Currently supported equations of state to model compressible or (nearly) incompressible fluids
  BETTER_ENUM(EquationOfState, char, ideal_gas, stiffened_gas, noble_abel_stiffened_gas)

  // Forward declarations
  namespace EOS
  {
    template <int dim, typename number>
    class EquationOfStateUtils;

    template <int dim, typename number>
    class IdealGas;

    template <int dim, typename number>
    class StiffenedGas;

    template <int dim, typename number>
    class NobleAbelStiffenedGas;
  } // namespace EOS

  template <typename number>
  struct EOSData
  {
    // type of equation of state
    EquationOfState type = EquationOfState::ideal_gas;

    // parameter to model molecular attraction within condensed matter
    // (required for stiffened_gas and noble_abel_stiffened_gas)
    number p_inf = std::numeric_limits<number>::max();

    // parameter to model the covolume of the fluid, i.e., the exclude volume
    // due to the finite size of the molecules
    // (required for noble_abel_stiffened_gas)
    number b = std::numeric_limits<number>::max();

    // parameter to model the 'heat bound', i.e., the energy due to chemical bounds,
    // hydrogen bondings, latent heat,...
    // (required for noble_abel_stiffened_gas)
    number q = std::numeric_limits<number>::min();

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("equation of state");
      {
        prm.add_parameter(
          "type",
          type,
          "Type of equation of state. "
          "The options are \"ideal_gas\", \"stiffened_gas\" and \"noble_abel_stiffened_gas\".",
          dealii::Patterns::Selection("ideal_gas|stiffened_gas|noble_abel_stiffened_gas"));
        prm.add_parameter(
          "p inf",
          p_inf,
          "Numerical EOS parameter to model the "
          "molecular attraction within condensed matter. The variable is required for the "
          "stiffened gas EOS and the Noble-Abel stiffened gas EOS. The minimum value is 0.",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter(
          "b",
          b,
          "Numerical EOS parameter to model the covolume "
          "of the fluid, i.e., the exclude volume due to the finite size of the molecules. "
          "The variable is required for the Noble-Abel stiffened gas EOS. "
          "The minimum value is 0.",
          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter(
          "q",
          q,
          "Numerical EOS parameter to model the "
          "'heat bound', i.e., the energy due to chemical bounds, hydrogen bondings, "
          "latent heat,.... The variable is required for the Noble-Abel stiffened gas EOS. The "
          "maximum value is 0.",
          dealii::Patterns::Double(std::numeric_limits<number>::min(), 0.));
      }
      prm.leave_subsection();
    };
  };

  template <typename number>
  struct CompressibleFluidMaterialPhaseData
  {
    // specific isobaric heat (SI: J/(kg K))
    number specific_isobaric_heat = 1000.0;

    // dynamic viscosity (SI: kg/(m s))
    number dynamic_viscosity = 1. / 1600.;

    // ratio of specific heat (specific heat at constant pressure divided by
    // specific heat at constant volume)
    number gamma = 1.4;

    // specific gas constant (SI: J/(kg K))
    number specific_gas_constant = 287.1;

    // reference density for interior penalty (SI: kg/m3)
    number reference_density = 1.0;

    // thermal conductivity (SI: W/(m K))
    number thermal_conductivity = std::numeric_limits<number>::max();

    // data for the equation of state
    EOSData<number> eos_data;

    void
    add_parameters(dealii::ParameterHandler &prm, const bool is_gas_phase)
    {
      const std::string subsection_name = is_gas_phase ? "gas" : "liquid";
      prm.enter_subsection("material");
      prm.enter_subsection(subsection_name);
      {
        prm.add_parameter("specific isobaric heat",
                          specific_isobaric_heat,
                          "Specific isobaric heat.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("dynamic viscosity",
                          dynamic_viscosity,
                          "Dynamic viscosity.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("gamma",
                          gamma,
                          "Isentropic exponent, i.e., ratio of specific heat (c_p/c_v).",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("specific gas constant", specific_gas_constant, "Specific gas constant.");
        prm.add_parameter("reference density",
                          reference_density,
                          "Reference density for computing the interior penalty factor. "
                          "A good first guess is to choose a value in the order of the fluid"
                          " density. If instabilities occur, the reference density can be "
                          "decreased, so that the symmetric interior penalization is increased.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("thermal conductivity",
                          thermal_conductivity,
                          "Thermal conductivity.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        eos_data.add_parameters(prm);
      }
      prm.leave_subsection();
      prm.leave_subsection();
    };

    void
    post(const bool is_gas)
    {
      // Set thermal conductivity, if not explicitly set by the user.
      // For physical consistency, set thermal conductivity based on the user-defined dynamic
      // viscosity, gamma and specific gas constant. The Prandtl number is currently set
      // constant to Pr=0.71 for the gas phase (air) and to Pr=0.01 for the liquid phase (metal).
      const number Pr = is_gas ? 0.71 : 0.01;
      if (thermal_conductivity == std::numeric_limits<number>::max())
        thermal_conductivity =
          dynamic_viscosity * gamma * specific_gas_constant / (gamma - 1.) * 1. / Pr;

      // Ensure that parameters are set for advanced equations of state
      if (eos_data.type == EquationOfState::stiffened_gas)
        AssertThrow(eos_data.p_inf != std::numeric_limits<number>::max(),
                    dealii::ExcMessage(
                      "The parameter p_inf is required for the stiffened gas EOS."));
      else if (eos_data.type == EquationOfState::noble_abel_stiffened_gas)
        AssertThrow(eos_data.p_inf != std::numeric_limits<number>::max() and
                      eos_data.b != std::numeric_limits<number>::max() and
                      eos_data.q != std::numeric_limits<number>::min(),
                    dealii::ExcMessage(
                      "The parameters p_inf, b and q are required for the Noble-Abel stiffened"
                      " gas EOS."));
    };
  };
} // namespace MeltPoolDG::Flow
