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
    , residual_tolerance(nlsolve_data.abs_residual_tolerance)
    , field_correction_tolerance(nlsolve_data.field_correction_tolerance)
    , nox_parameters(Teuchos::rcp(new Teuchos::ParameterList))
    , nox_additional_data(nlsolve_data.max_nonlinear_iterations +
                            nlsolve_data.max_nonlinear_iterations_alt,
                          nlsolve_data.abs_residual_tolerance,
                          nlsolve_data.rel_residual_tolerance)
    , nox_solver(nox_additional_data, nox_parameters)
  {
    set_nox_solver_parameters();
  }

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::reinit()
  {}

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::solve(VectorType &solution)
  {
    // compute the residual
    nox_solver.residual = [this](const VectorType &src, VectorType &dst) {
      try
        {
          this->template get_function<distribute_constraints_function_type>(
            NonlinearSolverFunctions::distribute_constraints)(const_cast<VectorType &>(src));
        }
      catch (...)
        {}
      this->template get_function<residual_function_type>(NonlinearSolverFunctions::residual)(src,
                                                                                              dst);
      dst *= -1.;
    };

    // setup jacobian (only required when matrix-based)
    try
      {
        nox_solver.setup_jacobian = this->template get_function<setup_jacobian_function_type>(
          NonlinearSolverFunctions::setup_jacobian);
      }
    catch (...)
      {
        nox_solver.setup_jacobian = [this](const auto &solution) {};
      }

    // solve with external defined jacobian (note: the variable tolerance is not used as the
    // tolerance is defined in the MeltPoolDG linear solver data)
    nox_solver.solve_with_jacobian_and_track_n_linear_iterations =
      [this](const VectorType             &rhs,
             VectorType                   &dst,
             [[maybe_unused]] const double tolerance) -> int {
      const int lin_iter = this->template get_function<solve_with_jacobian_function_type>(
        NonlinearSolverFunctions::solve_with_jacobian)(rhs, dst);
      return lin_iter;
    };

    // post-processing after each Newton iteration
    nox_solver.check_iteration_status = [this](const unsigned int nl_iter,
                                               const double       res_norm,
                                               const VectorType  &solution,
                                               const VectorType &) -> dealii::SolverControl::State {
      return dealii::SolverControl::iterate;
    };



    print_header();
    nox_solver.solve(solution);

    try
      {
        this->template get_function<distribute_constraints_function_type>(
          NonlinearSolverFunctions::distribute_constraints)(solution);
      }
    catch (...)
      {}

    if (nlsolve_data.verbosity_level >= 0)
      {
        std::ostringstream str_sol;
        str_sol << "Newton Raphson solver converged: ||solution|| = " << std::scientific
                << std::setprecision(5)
                << this->template get_function<norm_of_solution_vector_function_type>(
                     NonlinearSolverFunctions::norm_of_solution_vector)();

        Journal::print_line(pcout, str_sol.str(), "newton_raphson_solver");
      }
  }

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::print_header() const
  {
    if (nlsolve_data.verbosity_level >= 1)
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
    residual_tolerance         = nlsolve_data.abs_residual_tolerance_alt;
    field_correction_tolerance = nlsolve_data.field_correction_tolerance_alt;
  }

  template <typename VectorType>
  std::string
  NewtonRaphsonSolver<VectorType>::print_checkmark(const bool is_converged) const
  {
    return (is_converged) ? " ✓ " : " ✗ ";
  }

  template <typename VectorType>
  void
  NewtonRaphsonSolver<VectorType>::set_nox_solver_parameters()
  {
    // specify nonlinear solver type
    nox_parameters->set("Nonlinear Solver", "Line Search Based");
    // specify method of line search
    nox_parameters->sublist("Line Search").set("Method", "Full Step");
    // specify direction
    nox_parameters->sublist("Direction").set("Method", "Newton");
    // prevent output
    nox_parameters->sublist("Printing").set("Output Information", 0);
  }

  template class NewtonRaphsonSolver<dealii::LinearAlgebra::distributed::Vector<double>>;
} // namespace MeltPoolDG