#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  /**
   * Parameters for the nonlinear solver.
   */
  template <typename number = double>
  struct NonlinearSolverData
  {
    //! maximum number of nonlinear iterations
    int max_nonlinear_iterations = 10;
    //!
    number field_correction_tolerance = 1e-10;
    //! absolute tolerance of the residual
    number abs_residual_tolerance = 1e-9;
    //! relative tolerance of the residual
    number rel_residual_tolerance         = 1e-20;
    //! alternative maximum number of nonlinear iterations
    int    max_nonlinear_iterations_alt   = 0;
    //!
    number field_correction_tolerance_alt = 1e-9;
    //! alternative absolute tolerance of the residual
    number abs_residual_tolerance_alt     = 1e-8;

    int    verbosity_level                = -1;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG
