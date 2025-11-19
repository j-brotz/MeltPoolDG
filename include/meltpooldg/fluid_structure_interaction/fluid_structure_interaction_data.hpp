#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/fluid_structure_interaction/brinkman_penalization_data.hpp>
#include <meltpooldg/utilities/better_enum.hpp>

namespace MeltPoolDG
{
  BETTER_ENUM(FSICouplingMethod, char, brinkman_penalization, stokes_law);

  template <typename number>
  struct FluidStructureInteractionData
  {
    FSICouplingMethod fsi_coupling_method = FSICouplingMethod::brinkman_penalization;

    BrinkmanPenalizationData<number> brinkman_penalization_data;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("fluid structure interaction");
      {
        prm.add_parameter(
          "coupling method",
          fsi_coupling_method,
          "Type of FSI coupling method. Available options are 'brinkman_penalization' and 'stokes_law'.");
        brinkman_penalization_data.add_parameters(prm);
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
