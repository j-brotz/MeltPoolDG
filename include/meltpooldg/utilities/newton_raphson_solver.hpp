/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, March 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, typename VectorType = LinearAlgebra::distributed::Vector<double>>
  class NewtonRaphsonSolver
  {
  private:
    const ScratchData<dim> &           scratch_data;
    const NonlinearSolverData<double> &nlsolve_data;

    const unsigned int dof_idx;
    const unsigned int quad_idx;
    const VectorType & solution_old;
    VectorType &       solution;

    const int max_number_of_iterations;
    double    residual_tolerance;
    double    field_correction_tolerance;

    const std::function<void(VectorType &rhs)> &create_rhs;
    const std::function<int(VectorType &solution_update, const VectorType &rhs)>
      &solve_linear_system;


    VectorType rhs;
    VectorType solution_update;
    int        iteration_counter = 0;

  public:
    NewtonRaphsonSolver(const ScratchData<dim> &                    scratch_data,
                        const NonlinearSolverData<double> &         nlsolve_data,
                        const unsigned int                          dof_idx,
                        const unsigned int                          quad_idx,
                        const VectorType &                          solution_old,
                        VectorType &                                solution,
                        const std::function<void(VectorType &rhs)> &create_rhs,
                        const std::function<int(VectorType &solution_update, const VectorType &rhs)>
                          &solve_linear_system);

    void
    solve();

  public:
    double
    suggest_new_time_increment();

    void
    set_tolerances_to_alternative_values();

    bool
    is_converged();

    std::string
    print_checkmark(bool is_converged);

    void
    solve_increment();
  };

} // namespace MeltPoolDG
