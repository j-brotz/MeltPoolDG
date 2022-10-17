#pragma once

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  /**
   * Parameters for the nonlinear solver.
   */
  template <typename number = double>
  struct NonlinearSolverData
  {
    int    max_nonlinear_iterations       = 10;
    number field_correction_tolerance     = 1e-10;
    number residual_tolerance             = 1e-9;
    int    max_nonlinear_iterations_alt   = 0;
    number field_correction_tolerance_alt = 1e-9;
    number residual_tolerance_alt         = 1e-8;

    void
    add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("nlsolve");
      {
        prm.add_parameter(
          "max nonlinear iterations",
          max_nonlinear_iterations,
          "Set the number of maximum nonlinear iterations with standard tolerances.");
        prm.add_parameter(
          "field correction tolerance",
          field_correction_tolerance,
          "Set the tolerance for the maximum allowed correction of the unknown field.");
        prm.add_parameter(
          "residual tolerance",
          residual_tolerance,
          "Set the tolerance for the maximum allowed residual of the nonlinear system.");
        prm.add_parameter(
          "max nonlinear iterations alt",
          max_nonlinear_iterations_alt,
          "Set the number of maximum nonlinear iterations with alternative tolerances.");
        prm.add_parameter(
          "field correction tolerance alt",
          field_correction_tolerance_alt,
          "Set the alternative tolerance for the maximum allowed correction of the unknown field.");
        prm.add_parameter(
          "residual tolerance alt",
          residual_tolerance_alt,
          "Set the alternative tolerance for the maximum allowed residual of the nonlinear system.");
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
