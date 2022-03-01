/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, March 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, typename VectorType = LinearAlgebra::distributed::Vector<double>>
  class NewtonRaphsonSolver
  {
  public:
    std::function<void(const VectorType &evaluation_point, VectorType &residual)> residual = {};
    std::function<int(const VectorType &rhs, VectorType &dst)> solve_with_jacobian         = {};
    std::function<void(VectorType &v)>                         reinit_vector               = {};
    std::function<void(VectorType &v)>                         distribute_constraints      = {};

    // TODO: remove and replace by standard l2_norm() (?)
    std::function<double()> norm_of_solution_vector = {};

  private:
    const NonlinearSolverData<double> nlsolve_data;

    dealii::ConditionalOStream pcout;

    const int max_number_of_iterations;
    double    residual_tolerance;
    double    field_correction_tolerance;

    VectorType rhs;
    VectorType solution_update;

    int                iteration_counter = 0;
    int                linear_iter_acc   = 0;
    std::ostringstream str_;

  public:
    NewtonRaphsonSolver(const NonlinearSolverData<double> &nlsolve_data);

    void
    solve(VectorType &solution);

    const VectorType &
    get_residual() const
    {
      return rhs;
    }

    const VectorType &
    get_solution_update() const
    {
      return solution_update;
    }

  private:
    void
    solve_increment();

    double
    suggest_new_time_increment();

    void
    set_tolerances_to_alternative_values();

    bool
    is_converged();

    void
    print_header();

    std::string
    print_checkmark(bool is_converged);
  };

} // namespace MeltPoolDG
