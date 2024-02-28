#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

BETTER_ENUM(DarcyDampingFormulation, char, implicit_formulation, explicit_formulation)

namespace MeltPoolDG::Flow
{
  template <typename number = double>
  struct DarcyDampingData
  {
    number                  mushy_zone_morphology   = 0.0;
    number                  avoid_div_zero_constant = 1e-3;
    DarcyDampingFormulation formulation             = DarcyDampingFormulation::implicit_formulation;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };

} // namespace MeltPoolDG::Flow