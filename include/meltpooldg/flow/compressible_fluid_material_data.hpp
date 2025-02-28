#pragma once

#include <deal.II/base/parameter_handler.h>
#include <meltpooldg/utilities/better_enum.hpp>

namespace MeltPoolDG::Flow
{
  struct CompressibleFlowData;
  BETTER_ENUM(EOS, char, ideal_gas, stiffened_gas, noble_abel_stiffend_gas)

  template <typename number = double>
  struct EOSParameters
  {
    number p_inf = 0.;
    number b = 0.;
    number q = 0.;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  template <typename number = double>
  struct CompressibleFluidMaterialPhaseData
  {
    number thermal_conductivity   = 0.0;
    number specific_isobaric_heat = 0.0;
    number dynamic_viscosity      = 0.0;
    number gamma                  = 0.0;
    EOS equation_of_state = EOS::ideal_gas;

    EOSParameters<number> eos_parameters;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::Flow