#include <deal.II/numerics/data_out.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG
{
  template <typename VectorType>
  NewtonRaphsonSolver<VectorType>::NewtonRaphsonSolver(
    const NonlinearSolverData<double> &nlsolve_data)
    : nlsolve_data(nlsolve_data)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , max_number_of_iterations(nlsolve_data.max_nonlinear_iterations +
                               nlsolve_data.max_nonlinear_iterations_alt)
    , residual_tolerance(nlsolve_data.residual_tolerance)
    , field_correction_tolerance(nlsolve_data.field_correction_tolerance)
  {}

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::solve(VectorType &solution)
  {
    print_header();

    reinit_vector(rhs);
    reinit_vector(solution_update);

    int i           = 0;
    linear_iter_acc = 0;
    while (i < max_number_of_iterations)
      {
        solve_increment(solution);
        if (is_converged())
          {
            if (nlsolve_data.verbosity_level >= 1)
              {
                std::ostringstream str_sol;
                str_sol << "Newton Raphson solver converged: ||solution|| = " << std::scientific
                        << std::setprecision(5) << norm_of_solution_vector();

                Journal::print_line(pcout, str_sol.str(), "newton_raphson_solver");
              }

            {
              const ScopedName sc("nonlinear_solve");
              IterationMonitor<double>::add_linear_iterations(sc, i);
            }

            {
              const ScopedName sc("linear_solve_acc");
              IterationMonitor<double>::add_linear_iterations(sc, linear_iter_acc);
            }

            return;
          }

        solution += solution_update;
        distribute_constraints(solution);
        i++;
      }

    AssertThrow(is_converged(), ExcNewtonDidNotConverge());
  }

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::print_header() const
  {
    if (nlsolve_data.verbosity_level >= 2)
      {
        Journal::print_line(pcout);
        Journal::print_line(pcout, std::string(10, ' ') + std::string(60, '_'));
        std::ostringstream str;
        str << std::string(10, ' ') << std::setw(15) << "#lin solve" << std::internal
            << std::setw(15) << "||residual||" << std::internal << std::setw(15) << "||ddx||";
        Journal::print_line(pcout, str.str(), "newton_raphson_solver");
        Journal::print_line(pcout, std::string(10, ' ') + std::string(60, '_'));
      }
  }

  template <typename VectorType>
  double
  NewtonRaphsonSolver<VectorType>::suggest_new_time_increment()
  {
    AssertThrow(false, ExcNotImplemented());
    return 0.0;
  }

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::set_tolerances_to_alternative_values()
  {
    residual_tolerance         = nlsolve_data.residual_tolerance_alt;
    field_correction_tolerance = nlsolve_data.field_correction_tolerance_alt;
  }

  template <typename VectorType>
  bool
  NewtonRaphsonSolver<VectorType>::is_converged()
  {
    if (iteration_counter == nlsolve_data.max_nonlinear_iterations)
      set_tolerances_to_alternative_values();

    const double res_norm    = rhs.l2_norm();
    const double update_norm = solution_update.l2_norm();

    const bool residual_converged   = res_norm < residual_tolerance;
    const bool correction_converged = update_norm < field_correction_tolerance;

    if (nlsolve_data.verbosity_level >= 2)
      {
        str_ << std::right << std::setw(15) << std::scientific << std::setprecision(5) << res_norm
             << print_checkmark(residual_converged);
        str_ << std::right << std::setw(15) << std::scientific << std::setprecision(5)
             << update_norm << print_checkmark(correction_converged);

        Journal::print_line(pcout, str_.str(), "", 4);
        str_.str("");
      }

    return residual_converged && correction_converged;
  }

  template <typename VectorType>
  std::string
  NewtonRaphsonSolver<VectorType>::print_checkmark(const bool is_converged) const
  {
    return (is_converged) ? " ✓ " : " ✗ ";
  }

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::solve_increment(const VectorType &current_solution)
  {
    rhs             = 0.0;
    solution_update = 0.0;

    // compute residual
    residual(current_solution, rhs);
    const int iter = solve_with_jacobian(rhs, solution_update);
    {
      const ScopedName sc("linear_solve");
      IterationMonitor<double>::add_linear_iterations(sc, iter);
    }
    linear_iter_acc += iter;

    str_ << std::string(10, ' ') << std::right << std::setw(15) << std::setprecision(0) << iter;
  }

  template class NewtonRaphsonSolver<LinearAlgebra::distributed::Vector<double>>;
} // namespace MeltPoolDG
