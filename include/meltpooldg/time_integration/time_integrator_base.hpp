/**
 * @brief Time integrator base class from which all explicit time integrator classes are derived.
 */

#pragma once
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

#include <functional>

namespace MeltPoolDG
{
  using namespace dealii;

  /**
   * Requirements for a pde operator in order to use it with the explicit time integrator.
   */
  template <typename Operator, typename number, typename VectorType>
  concept ExplicitPDEOperator =
    requires(Operator                                               pde_operator,
             VectorType                                            &vec,
             const VectorType                                      &const_vec,
             number                                                 scalar,
             const std::function<void(unsigned int, unsigned int)> &func) {
      /**
       * Given an ODE of the form y' = f(y), this function computes f(y) based on the provided
       * parameters. It takes the current time, the current value of y, a destination vector where
       * f(y) will be stored, and an additional function that is applied after f(y) is computed.
       * This function is designed to be directly called from the matrix-free cell-loop and
       * therefore takes a pair of indices indicating the start and end range for processing.
       */
      pde_operator.apply_operator(scalar, vec, const_vec, func);
    };


  template <typename number, typename PDEOperator>
  class TimeIntegratorBase
  {
    using VectorType = LinearAlgebra::distributed::Vector<number>;

  public:
    explicit TimeIntegratorBase(const TimeIntegratorData &time_integrator_data_in)
      : time_integrator_data(time_integrator_data_in)
    {}

    virtual ~TimeIntegratorBase() = default;

    /**
     * This function returns the number of previous solutions required. Note that a corresponding
     * solution history must be of the size previous_solutions_required+1 to account for the current
     * solution.
     */
    virtual unsigned int
    required_solution_history_size() const = 0;

    /**
     * Sets up the necessary internal data structures.
     *
     * @param vector_template Reference vector used to define the partitioning for all internal
     * vectors.
     */
    virtual void
    reinit(const VectorType &vector_template) = 0;

    /**
     * As above, sets up the necessary internal data structures.
     */
    virtual void
    reinit(const ::TimeIntegration::SolutionHistory<VectorType> &solution_history)
    {
      reinit(solution_history.get_current_solution());
    }

    /**
     * Perform the actual time integration for a single time step.
     *
     * @param pde_operator Class providing the 'apply_operator()' function to compute f(y).
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     */
    virtual void
    perform_time_step(
      const PDEOperator                                                   &pde_operator,
      number                                                               current_time,
      number                                                               time_step,
      ::TimeIntegration::SolutionHistory<VectorType>                      &solution_history,
      const std::function<void(number, VectorType &, const VectorType &)> &pre_processing = {},
      const std::function<void(number, VectorType &, const VectorType &)> &post_processing =
        {}) = 0;

    /**
     * Function that returns the time integration scheme of the time integrator object.
     */
    TimeIntegratorSchemes
    get_integrator_type() const
    {
      return time_integrator_data.integrator_type;
    }

    /**
     * This function stores a pointer to the passed vector which can then be used in the derived
     * time integrator classes for providing additional data during the time integration. Which kind
     * of data is provided to this vector depends on the individual time integrator class.
     */
    void
    set_monitoring_vector(VectorType &monitoring_vector_in)
    {
      monitoring_vector = &monitoring_vector_in;
    }

    /**
     * This function sets the monitoring vector to nullptr in the case monitoring is no longer
     * needed.
     */
    void
    reset_monitoring_vector()
    {
      monitoring_vector = nullptr;
    }

  protected:
    const TimeIntegratorData time_integrator_data;
    VectorType              *monitoring_vector = nullptr;
  };
} // namespace MeltPoolDG
