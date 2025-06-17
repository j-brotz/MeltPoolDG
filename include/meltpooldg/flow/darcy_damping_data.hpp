#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

BETTER_ENUM(DarcyDampingFormulation, char, implicit_formulation, explicit_formulation)

namespace MeltPoolDG::Flow
{
  /**
   * @brief Collection of parameters for Darcy damping.
   */
  template <typename number>
  struct DarcyDampingData
  {
    /// Mushy zone morphology for Darcy damping
    number mushy_zone_morphology = 0.0;

    /// Parameter to avoid division by zero in the Kozeny–Carman equation for the Darcy damping
    /// force.
    number avoid_div_zero_constant = 1e-3;

    /// Formulation of the Darcy damping force
    DarcyDampingFormulation formulation = DarcyDampingFormulation::implicit_formulation;

    /**
     * @brief Add darcy damping parameters in the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::Flow
