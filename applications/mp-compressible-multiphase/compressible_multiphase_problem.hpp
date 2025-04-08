#pragma once

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/compressible_flow_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include "compressible_multiphase_case.hpp"

namespace MeltPoolDG::Multiphase
{
  template <int dim, typename number>
  class CompressibleMultiphaseProblem
  {
  private:
    using CaseType   = CompressibleMultiphaseCase<dim, number>;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    explicit CompressibleMultiphaseProblem(std::unique_ptr<CaseType> simulation_case)
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
    output_results(unsigned int time_step, number current_time);

    /**
     * Interpolates the values of a (currently) analytically given level-set function to the
     * level-set dof vector.
     */
    void
    interpolate_initial_level_set();

    /**
     * Do level-set advection and reinitialization.
     */
    void
    update_level_set();

    std::shared_ptr<CaseType> simulation_case;

    dealii::DoFHandler<dim>                                  dof_handler;
    dealii::DoFHandler<dim>                                  dof_handler_level_set;
    dealii::AffineConstraints<number>                        constraints;
    dealii::AffineConstraints<number>                        constraints_level_set;
    std::shared_ptr<ScratchData<dim, dim, number>>           scratch_data;
    std::shared_ptr<TimeIterator<number>>                    time_iterator;
    MeltPoolDG::Flow::CompressibleFlowOperation<dim, number> comp_multiphase_operation;
    std::unique_ptr<Profiling::ProfilingMonitor<number>>     profiling_monitor;

    unsigned int comp_multiphase_dof_idx{};
    unsigned int level_set_dof_idx{};
    unsigned int comp_multiphase_quad_idx{};

    std::unique_ptr<Postprocessor<dim, number>> post_processor;

    std::shared_ptr<dealii::Function<dim>> level_set_field_function;
    VectorType                             level_set;
  };

} // namespace MeltPoolDG::Multiphase
