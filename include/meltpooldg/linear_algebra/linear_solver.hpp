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
      const bool monitor_history = data.monitor_type != LinearSolverMonitorType::none;

      ReductionControl solver_control(data.max_iterations, data.abs_tolerance, data.rel_tolerance);

      if (monitor_history)
        solver_control.enable_history_data();

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

<<<<<<< HEAD
      solution.update_ghost_values();
=======
      solution.update_ghost_values(); // PM: why the hack do we call update ghost value here?

      if (solver_control.last_step() > 0 && monitor_history)
        {
          const auto &history_data = solver_control.get_history_data();

          const auto print_value = [](const unsigned int iteration, const double value) {
            std::cout << iteration << " " << value << std::endl;
          };

          if (data.monitor_type == LinearSolverMonitorType::all)
            {
              for (unsigned int i = 0; i < history_data.size(); ++i)
                {
                  print_value(i, history_data[i]);
                }
            }
          else if (data.monitor_type == LinearSolverMonitorType::reduced)
            {
              print_value(0, history_data[0]);
              if (history_data.size() >= 2)
                print_value(history_data.size() - 1, history_data.back());
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }

>>>>>>> 46b7fce7 (Monitor history of linear solvers)
      return solver_control.last_step();
    }
  };
} // namespace MeltPoolDG
