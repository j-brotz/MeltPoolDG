#pragma once
// for distributed vectors/matrices
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
// solvers
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

// preconditioner
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>

using namespace dealii;

namespace MeltPoolDG
{
  class LinearSolver
  {
  public:
    template <typename VectorType,
              typename OperatorType       = TrilinosWrappers::SparseMatrix,
              typename PreconditionerType = PreconditionIdentity>
    static int
    solve(const OperatorType &            system_matrix,
          VectorType &                    solution,
          const VectorType &              rhs,
          const LinearSolverData<double> &data,
          const PreconditionerType &      preconditioner = PreconditionIdentity())
    {
      ReductionControl solver_control(data.max_iterations, data.abs_tolerance, data.rel_tolerance);

      switch (data.solver_type)
        {
            case (LinearSolverType::CG): {
              auto solver = SolverCG<VectorType>(solver_control);
              solver.solve(system_matrix, solution, rhs, preconditioner);
              break;
            }
            case (LinearSolverType::GMRES): {
              auto solver = SolverGMRES<VectorType>(solver_control);
              solver.solve(system_matrix, solution, rhs, preconditioner);
              break;
            }
          default:
            AssertThrow(false, ExcNotImplemented());
        }

      solution.update_ghost_values();
      return solver_control.last_step();
    }
  };
} // namespace MeltPoolDG
