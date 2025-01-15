/**
 *@brief Implementation of a Newton-Rapson solver using the Trilinos NOX solver class.
 */

#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_base.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <deal.II/trilinos/nox.h>

namespace MeltPoolDG
{

  template <typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>>
  class NewtonRaphsonSolver final : public NonlinearSolverBase<VectorType>
  {
  public:
    typedef std::function<void(const VectorType &evaluation_point, VectorType &residual)>
      residual_function_type;
    typedef std::function<int(const VectorType &rhs, VectorType &dst)>
                                               solve_with_jacobian_function_type;
    typedef std::function<void(VectorType &v)> reinit_vector_function_type;
    typedef std::function<void(VectorType &v)> distribute_constraints_function_type;

    // TODO: remove and replace by standard l2_norm() (?)
    typedef std::function<double()>                 norm_of_solution_vector_funtion_type;
    typedef std::function<void(const VectorType &)> setup_jacobian_function_type;

    explicit NewtonRaphsonSolver(const NonlinearSolverData<double> &nlsolve_data);

    /**
     * Solve the equation system using the passed functions.
     *
     * @param solution The vector in which the solution will be stored.
     *
     * @throws This function throws an exception if the Newton solver does not converge.
     */
    void
    solve(VectorType &solution) override;

    /**
     * Initialize the NOX solver, i.e. set the internal NOX solver functions used to perform the
     * Newton step. After this function was called the class is in a state allowing to call the
     * solve() function.
     */
    void
    reinit() override;

    const VectorType &
    get_residual() const
    {
      return rhs;
    }

  private:
    void
    solve_increment();

    double
    suggest_new_time_increment();

    void
    set_tolerances_to_alternative_values();

    void
    print_header() const;

    std::string
    print_checkmark(bool is_converged) const;

    /**
     * This function sets the parameters for the trilinos nox solver such that the solve() function
     * performs a full newton step.
     */
    void
    set_nox_solver_parameters();

    bool
    is_converged();


    const NonlinearSolverData<double> nlsolve_data;

    const dealii::ConditionalOStream pcout;

    const int max_number_of_iterations;
    double    residual_tolerance;
    double    field_correction_tolerance;

    VectorType rhs;
    VectorType solution_update;

    int iteration_counter = 0;
    int linear_iter_acc   = 0;

    //! Trilinos nonlinear solver objects
    /*
    Teuchos::RCP<Teuchos::ParameterList>                                     nox_parameters;
    typename dealii::TrilinosWrappers::NOXSolver<VectorType>::AdditionalData nox_additional_data;
    dealii::TrilinosWrappers::NOXSolver<VectorType>                          nox_solver;
*/

    std::ostringstream str_;
  };

} // namespace MeltPoolDG
