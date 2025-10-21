#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/better_enum.hpp>

BETTER_ENUM(FSICouplingMethod, char, brinkman_penalty, unresolved_penalty);

template <typename number>
struct FluidStructureInteractionData
{
  FSICouplingMethod fsi_coupling_method = FSICouplingMethod::brinkman_penalty;
  number            permeability{};

  void
  add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("brinkman penalization");
    {
      prm.add_parameter("coupling method", fsi_coupling_method);
      prm.add_parameter("permeability", permeability);
    }
    prm.leave_subsection();
  }
};