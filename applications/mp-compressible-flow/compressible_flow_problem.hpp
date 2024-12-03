#pragma once

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/compressible_flow_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include "compressible_flow_case.hpp"

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim>
  class CompressibleFlowProblem
  {
  private:
    using CaseType   = CompressibleFlowCase<dim>;
    using VectorType = LinearAlgebra::distributed::Vector<double>;

  public:
    CompressibleFlowProblem(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    /**
     * Run the simulation. This function internally sets up all required member data and performs
     * the time loop.
     */
    void
    run();

  private:
    /**
     * Set up all dof related data.
     */
    void
    setup_dof_system();

    /**
     * Initializes the relevant member data required for solving the compressible flow problem
     * and prepares the object for starting the time loop. This includes setting the necessary
     * boundary and initial conditions.
     *
     * @note Internally calls the setup_dof_system() function.
     */
    void
    initialize();

    /**
     *  Perform the output of the results.
     */
    void
    output_results(unsigned int time_step, double current_time);


    std::unique_ptr<CaseType> simulation_case;

    DoFHandler<dim>                                         dof_handler;
    AffineConstraints<double>                               constraints;
    DoFHandler<dim>                                         dof_handler_velocity;
    std::shared_ptr<ScratchData<dim>>                       scratch_data;
    std::unique_ptr<TimeIterator<double>>                   time_iterator;
    std::unique_ptr<CompressibleFlowOperation<dim, double>> comp_flow_operation;
    std::unique_ptr<Profiling::ProfilingMonitor<double>>    profiling_monitor;


    unsigned int comp_flow_dof_idx;
    unsigned int comp_flow_quad_idx;

    std::unique_ptr<Postprocessor<dim>> post_processor;
  };

} // namespace MeltPoolDG::Flow
