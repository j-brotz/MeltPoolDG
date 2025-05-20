#pragma once

#include <deal.II/base/parameter_handler.h>

template <typename number>
struct BrinkmanPenalizationData
{
  number permeability;
  number porosity;

  void
  add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Brinkman penalization");
    {
      prm.add_parameter("permeability", permeability);
      prm.add_parameter("porosity", porosity);
    }
    prm.leave_subsection();
  }
};