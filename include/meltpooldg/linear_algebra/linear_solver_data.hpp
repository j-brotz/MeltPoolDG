#pragma once

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  BETTER_ENUM(PreconditionerType,
              char,
              // No preconditioner.
              Identity,
              // Algebraic multigrid preconditioner from the Trilinos package ...
              AMG,
              // ... potentially with reduced effort in computing the system matrix, e.g., by
              // neglecting face integrals.
              AMGReduced,
              // Incomplete LU factorization preconditioner from the Trilinos package ...
              ILU,
              // ... potentially with reduced effort in computing the system matrix, e.g., by
              // neglecting face integrals.
              ILUReduced,
              // Use the inverse diagonal of the system matrix as preconditioner ...
              Diagonal,
              // ... potentially with reduced effort in computing the system matrix, e.g., by
              // neglecting face integrals.
              DiagonalReduced)

  BETTER_ENUM(LinearSolverType,
              char,
              // conjugate-gradient
              CG,
              // generalized minimal residual
              GMRES)

  BETTER_ENUM(LinearSolverMonitorType,
              char,
              // do not monitor history (production run)
              none,
              // print first and last residual
              reduced,
              // print full history
              all)

  // choose the particular predictor type for the nonlinear/linear solver
  BETTER_ENUM(PredictorType,
              char,
              // no predictor specified; use old value as initial guess
              none,
              // calculate the predictor by a linear combination from the two old solution vectors
              linear_extrapolation)

  /**
   * Parameters for the linear solver.
   */
  template <typename number = double>
  struct LinearSolverData
  {
    bool               do_matrix_free      = true;
    PreconditionerType preconditioner_type = PreconditionerType::Identity;
    LinearSolverType   solver_type         = LinearSolverType::GMRES;
    unsigned int       max_iterations      = 10000;
    number             rel_tolerance       = 1e-12;
    number             abs_tolerance       = 1e-50;
    PredictorType      predictor           = PredictorType::none;

    LinearSolverMonitorType monitor_type = LinearSolverMonitorType::none;

    void
    add_parameters(ParameterHandler &prm);
  };

} // namespace MeltPoolDG
