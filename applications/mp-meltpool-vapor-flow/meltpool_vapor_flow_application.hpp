#pragma once

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

#include <memory>

#include "meltpool_vapor_flow_case.hpp"

namespace MeltPoolDG::MeltPoolVaporFlow
{
  /**
   * @brief Application for the simulation of the coupled meltpool and vapor flow problem where the
   * vapor flow is modeled as a compressible flow.
   */
  template <int dim, typename number>
  class Application
  {
  public:
    using CaseType = Case<dim, number>;

    /**
     * @brief Constructor.
     *
     * @param simulation_case Pointer to the considered simulation case.
     */
    explicit Application(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    /**
     * Run the actual simulation. This function internally sets up all required member data and
     * performs the time loop.
     */
    void
    run();

  private:
    /// Pointer to the considered simulation case
    std::shared_ptr<CaseType> simulation_case;

    /// Pointer to the time iterator
    std::shared_ptr<TimeIntegration::TimeIterator<number>> time_iterator;

    /// Pointer to the profiling object
    std::unique_ptr<Profiling::ProfilingMonitor<number>> profiling_monitor;

    /// Pointer to the post processor object
    std::unique_ptr<Postprocessor<dim, number>> post_processor;

    /**
     * Initializes the relevant member data required for solving the coupled meltpool-vapor
     * flow problem and prepares the object for starting the time loop. This includes setting the
     * necessary boundary and initial conditions.
     */
    void
    initialize();

    /**
     * Perform the output of the results, i.e., do relevant post processing step including printing
     * any relevant information to the console and writing paraview files.
     *
     * @param time_step Current time step size.
     * @param current_time Current time at t^n.
     * @param force_output If true, output will be generated regardless of the output frequency
     * defined in the parameters.
     */
    void
    output_results(unsigned int time_step, number current_time, const bool force_output = false);
  };

} // namespace MeltPoolDG::MeltPoolVaporFlow
