#pragma once
// for distributed vectors/matrices
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
// solvers
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

// preconditioner
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <meltpooldg/interface/parameters.hpp>

using namespace dealii;

namespace MeltPoolDG
{
  class LinearSolve
  {
  public:
    template <typename VectorType,
              typename SolverType         = SolverGMRES<VectorType>,
              typename OperatorType       = TrilinosWrappers::SparseMatrix,
              typename PreconditionerType = PreconditionIdentity>
    static int
    solve(const OperatorType &      system_matrix,
          VectorType &              solution,
          const VectorType &        rhs,
          const double              rel_tolerance  = 1e-12,
          const unsigned int        max_iterations = 10000,
          const PreconditionerType &preconditioner = PreconditionIdentity())
    {
      ReductionControl solver_control(max_iterations, 1e-50, rel_tolerance);

      SolverType solver(solver_control);

      solver.solve(system_matrix, solution, rhs, preconditioner);

      solution.update_ghost_values();
      return solver_control.last_step();
    }
  };
} // namespace MeltPoolDG
