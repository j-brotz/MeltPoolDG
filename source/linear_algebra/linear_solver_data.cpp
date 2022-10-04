#include <meltpooldg/linear_algebra/linear_solver_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  LinearSolverData<number>::add_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("linear solver");
    {
      prm.add_parameter("solver type",
                        this->solver_type,
                        "Set this parameter for choosing an iterative linear solver type.");
      prm.add_parameter("preconditioner type",
                        this->preconditioner_type,
                        "Set this parameter for choosing a preconditioner type.");
      prm.add_parameter(
        "max iterations",
        this->max_iterations,
        "Set the maximum number of iterations for solving the linear system of equations.");
      prm.add_parameter(
        "rel tolerance",
        this->rel_tolerance,
        "Set the relative tolerance for a successful solution of the linear system of equations.");
      prm.add_parameter(
        "abs tolerance",
        this->abs_tolerance,
        "Set the absolute tolerance for a successful solution of the linear system of equations.");
      prm.add_parameter(
        "do matrix free",
        this->do_matrix_free,
        "Set this parameter if a matrix free solution procedure should be performed.");
      prm.add_parameter("monitor type",
                        this->monitor_type,
                        "Set the monitor type of the linear solver.");
    }
    prm.leave_subsection();
  }

  template struct LinearSolverData<double>;
} // namespace MeltPoolDG
