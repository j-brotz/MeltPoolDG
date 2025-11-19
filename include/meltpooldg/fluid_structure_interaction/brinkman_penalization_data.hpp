#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_util.hpp>


namespace MeltPoolDG
{
  template <typename number>
  struct BrinkmanPenalizationData
  {
    number permeability{};

    MaskFunctionType mask_function_type = MaskFunctionType::discontinuous;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("brinkman penalization");
      {
        prm.add_parameter("permeability",
                          permeability,
                          "Permeability used in the Brinkman penalization term.");
        prm.add_parameter(
          "mask type",
          mask_function_type,
          "Type of mask function used for Brinkman penalization. Available options are 'discontinuous'.");
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
