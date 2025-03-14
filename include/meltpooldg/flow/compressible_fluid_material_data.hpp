#pragma once

#include <deal.II/base/parameter_handler.h>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

namespace MeltPoolDG::Flow
{
  BETTER_ENUM(EOS, char, ideal_gas, stiffened_gas, noble_abel_stiffened_gas)

  template <typename number = double>
  struct EOSParameters
  {
    // parameter to model molecular attraction within condensed matter
    // (required for stiffened_gas and noble_abel_stiffened_gas)
    number p_inf = dealii::numbers::invalid_double;

    // parameter to model the covolume of the fluid, i.e., the exclude volume
    // due to the finite size of the molecules
    // (required for noble_abel_stiffened_gas)
    number b = dealii::numbers::invalid_double;

    // parameter to model the 'heat bound', i.e., the energy due to chemical bounds,
    // hydrogen bondings, latent heat,...
    // (required for noble_abel_stiffened_gas)
    number q = dealii::numbers::invalid_double;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("EOS parameters");
      {
        prm.add_parameter("p inf", p_inf, "Numerical EOS parameter.");
        prm.add_parameter("b", b, "Numerical EOS parameter.");
        prm.add_parameter("q", q, "Numerical EOS parameter.");
      }
      prm.leave_subsection();
    };
  };

  template <typename number = double>
  struct CompressibleFluidMaterialPhaseData
  {
    // specific isobaric heat (SI: J/(kg K))
    number specific_isobaric_heat = 1000.0;

    // dynamic viscosity (SI: kg/(m s))
    number dynamic_viscosity      = 1./1600.;

    // ratio of specific heat (specific heat at constant pressure divided by
    // specific heat at constant volume)
    number gamma                  = 1.4;

    // specific gas constant (SI: J/(kg K))
    number specific_gas_constant  = 287.1;

    // reference density for interior penalty
    number reference_density      = 1.0;

    // thermal conductivity (SI: W/(m K)) (default definition with Prandtl number Pr=0.71)
    number thermal_conductivity   = dynamic_viscosity * gamma * specific_gas_constant / (gamma - 1.) * 1 / 0.71;

    // equation of state
    EOS equation_of_state = EOS::ideal_gas;

    EOSParameters<number> eos_parameters;

    void
    add_parameters(dealii::ParameterHandler &prm, const bool is_gas_phase)
    {
      const std::string subsection_name = is_gas_phase ? "material data gas phase" : "material data liquid phase";
      prm.enter_subsection(subsection_name);
      {
        prm.add_parameter("specific isobaric heat", specific_isobaric_heat, "Specific isobaric heat.");
        prm.add_parameter("dynamic viscosity", dynamic_viscosity, "Dynamic viscosity.");
        prm.add_parameter("gamma", gamma, "Isentropic exponent, i.e., ratio of specific heat (c_p/c_v).");
        prm.add_parameter("specific gas constant", specific_gas_constant, "Specific gas constant.");
        prm.add_parameter("reference density", reference_density,"Reference density for computing the interior penalty factor.");
        prm.add_parameter("thermal conductivity", thermal_conductivity, "Thermal conductivity.");
        prm.add_parameter("equation of state",equation_of_state,"Equation of state.");
        eos_parameters.add_parameters(prm);
      }
      prm.leave_subsection();
    };
  };
} // namespace MeltPoolDG::Flow