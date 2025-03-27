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
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

  public:
    explicit CompressibleFlowProblem(std::unique_ptr<CaseType> simulation_case)
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
     *
     *  @param time_step Current time step size.
     *  @param current_time Current time at t^n.
     */
    void
    output_results(unsigned int time_step, double current_time);

    /**
     * Interpolates the values of a (currently) analytically given level-set function to the
     * level-set dof vector.
     */
    void
    compute_level_set();

    std::shared_ptr<CaseType> simulation_case;

    dealii::DoFHandler<dim>                              dof_handler;
    dealii::DoFHandler<dim>                              dof_handler_level_set;
    dealii::AffineConstraints<double>                    constraints;
    dealii::AffineConstraints<double>                    constraints_level_set;
    std::shared_ptr<ScratchData<dim>>                    scratch_data;
    std::shared_ptr<TimeIterator<double>>                time_iterator;
    CompressibleFlowOperation<dim, double>               comp_flow_operation;
    std::unique_ptr<Profiling::ProfilingMonitor<double>> profiling_monitor;

    unsigned int comp_flow_dof_idx{};
    unsigned int level_set_dof_idx{};
    unsigned int comp_flow_quad_idx{};

    std::unique_ptr<Postprocessor<dim, double>> post_processor;

    std::shared_ptr<dealii::Function<dim>> level_set_field_function;
    VectorType                             level_set;
  };

} // namespace MeltPoolDG::Flow
