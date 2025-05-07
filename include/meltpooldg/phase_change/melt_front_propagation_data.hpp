#pragma once

#include <deal.II/base/parameter_handler.h>

template <typename number>
struct MeltFrontPropagationData
{
  bool   set_velocity_to_zero       = false;
  bool   do_not_reinitialize        = false;
  number solid_fraction_lower_limit = 1.0;

  void
  add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("melt front propagation");
    {
      prm.add_parameter(
        "set velocity to zero in solid",
        set_velocity_to_zero,
        "Set this parameter to true to constrain the flow velocity in the solid domain.");
      prm.add_parameter(
        "do not reinitialize in solid",
        do_not_reinitialize,
        "Set this parameter to true to forbid reinitialization of the level set field the solid domain.");
      prm.add_parameter(
        "solid fraction lower limit",
        solid_fraction_lower_limit,
        "Lower limit of the solid fraction for where the flow velocity / level set is "
        "set to zero if \"mp set velocity to zero in solid\" or \"mp set level set to zero in solid\" "
        "are enabled.",
        dealii::Patterns::Double(0.0, 1.0));
    }
    prm.leave_subsection();
  }
};
