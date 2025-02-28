#pragma once

#include <meltpooldg/flow/compressible_fluid_material_data.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  void
  EOSParameters<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("ghost-penalty");
    {
      prm.add_parameter("p inf", p_inf, "Numerical EOS parameter.");
      prm.add_parameter("b", b, "Numerical EOS parameter.");
      prm.add_parameter("q", q, "Numerical EOS parameter.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  CompressibleFluidMaterialPhaseData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("compressible fluid material phase data");
    {
      prm.add_parameter("thermal conductivity", thermal_conductivity, "Thermal conductivity.");
      prm.add_parameter("specific isobaric heat", specific_isobaric_heat, "Specific isobaric heat.");
      prm.add_parameter("dynamic viscosity", dynamic_viscosity, "Dynamic viscosity.");
      prm.add_parameter("gamma", gamma, "Isentropic exponent.");
      eos_parameters.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template struct MeltPoolDG::Flow::EOSParameters<double>;
  template struct MeltPoolDG::Flow::CompressibleFluidMaterialPhaseData<double>;
} // namespace MeltPoolDG::Flow
