#pragma once
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/time_integration/time_integration_setup.hpp>
#include <meltpooldg/utilities/solution_history.hpp>



namespace MeltPoolDG
{
  using namespace dealii;

  /**
   * Interface for time integration schemes
   * */
  template <int dim, typename Number = double>
  class TimeIntegrationBase
  {
    using VectorType = LinearAlgebra::distributed::Vector<double>;

  public:
    /**
     * @param old_time
     * @param time_step size of the current time step
     * @param solution_history object holding the advection field at different time instances.
     * @param rhs right hand side vector for implicit time integration schemes
     */
    virtual void
    perform_time_step(
      [[maybe_unused]] const double                                  old_time,
      [[maybe_unused]] const double                                 &time_step,
      [[maybe_unused]] TimeIntegration::SolutionHistory<VectorType> &solution_history,
      [[maybe_unused]] VectorType                                   &rhs) const = 0;



    virtual void
    reinit() = 0;
  };
} // namespace MeltPoolDG
