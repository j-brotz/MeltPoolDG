/**
 * @brief Base class for nonlinear solvers providing a generalized API.
 */

#pragma once

#include <deal.II/base/exceptions.h>

#include <any>
#include <unordered_map>

#include "newton_raphson_solver.hpp"

enum NonlinearSolverFunctions
{
  reinit_vector,
  solve_with_jacobian,
  residual,
  distribute_constraints,
  norm_of_solution_vector
};

template <typename VectorType>
class NonlinearSolverBase
{
public:
  NonlinearSolverBase() = default;

  virtual ~
  NonlinearSolverBase() = default;

  /**
   * Reinit function to set up the internal data structures and prepare the call to the solve()
   * function.
   */
  virtual void
  reinit() = 0;

  /**
   * Solve the equation system. The exact implementation depends on the individual nonlinear solver
   * classes.
   *
   * @param solution Vector in which the solution of the equation shall be stored.
   */
  virtual void
  solve(VectorType &solution) = 0;

  /**
   * In order to call the solve function the nonlinear solver functions (e.g. compute_jacobian())
   * must be set beforehand. This is the purpose of this function which stores the passed function
   * internally such that the nonlinear solver can call them later on.
   *
   * @param function_name Name of the function. For details see the corresponding nonlinear solver
   * class.
   * @param function The function corresponfing to the passed function name.
   *
   * @note This function does not check whether the passed function type agrees with the expected
   * function type by the nonlinear solver. If any of these checks ever happen depends on the
   * individual nonlinear solver implementation.
   */
  template <typename Function>
  void
  set_function(const NonlinearSolverFunctions function_name, const Function &function)
  {
    solver_functions[function_name] = function;
  }

protected:
  /**
   * This function returns a reference to the function stored in
   * @p solver_functions having the corresponding key passed to the function.
   * The usual use case is that after the user has set all relevant functions
   * with the @f set_function() the indivual derived nonlinear solver classes
   * use this function to set up the solver and solve the equation system.
   *
   * @param function_name Key of the function which shall be returned.
   *
   * @return The function corresponding to the passed key.
   * @throws An exception if the function cannot be fined in the member @p
   * solver_functions.
   */
  template <typename Function>
  Function &
  get_function(const NonlinearSolverFunctions function_name)
  {
    const auto it = solver_functions.find(function_name);
    AssertThrow(
      it != solver_functions.end(),
      dealii::ExcMessage(
        "The required function for the nonlinear solver can not be found. Have you set all functions?"));
    return std::any_cast<Function &>(it->second);
  }

private:
  //! A map to functions required by the corresponding nonlinear solver. To set these functions @f
  //! set_function() must be called with the corresponding function key of the type
  //! NonlinearSolverFunctions.
  std::unordered_map<NonlinearSolverFunctions, std::any> solver_functions;
};