#pragma once

#include "meltpooldg/flow/dg_compressible_flow_operation.hpp"
#include <meltpooldg/flow/compressible_flow_operation.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/amr_indicators.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

#include <memory>
#include <utility>

#include "cfd_dem_case.hpp"

namespace MeltPoolDG
{
  template <int dim, typename number>
  class CfdDemApplication
  {
  private:
    using CaseType   = CfdDemCase<dim, number>;
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

  public:
    explicit CfdDemApplication(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    /**
     * @brief Run the simulation. This function internally sets up all required member data and
     * performs the time loop.
     */
    void
    run();

  private:
    /**
     * @brief Set up all dof related data.
     */
    void
    setup_dof_system();

    /**
     * @brief Sets up the amr indicator used to identify cells which need to be refined/coarsened.
     *
     * Currently the indicator is an error estimator based on the combination of the jump and ssed
     * estimator as discussed in the conference paper
     *
     * F. Basile et al, A high-order h-adaptive discontinuous Galerkin method for unstructured grids
     * based on a posteriori error estimation. https://arc.aiaa.org/doi/10.2514/6.2021-1696
     */
    void
    setup_amr_indicator();

    /**
     * @brief Initializes the relevant member data required for solving the compressible flow
     * problem and prepares the object for starting the time loop. This includes setting the
     * necessary boundary and initial conditions.
     *
     * @note Internally calls the setup_dof_system() function.
     */
    void
    initialize();

    /**
     *  @brief Perform the output of the results.
     *
     *  @param time_step Current time step size.
     *  @param current_time Current time at t^n.
     */
    void
    output_results(unsigned int time_step, number current_time);

    /**
     * Check if mesh refinement is required by using the indicator amr_indicator and the bulk
     * criteria. If refinement is required this function also performs the refinement and performs
     * the solution transfer.
     */
    void
    refine_mesh();

    std::shared_ptr<CaseType> simulation_case;

    dealii::DoFHandler<dim>                                                     dof_handler;
    dealii::AffineConstraints<number>                                           constraints;
    std::shared_ptr<ScratchData<dim, dim, number>>                              scratch_data;
    std::shared_ptr<TimeIntegration::TimeIterator<number>>                      time_iterator;
    std::unique_ptr<Flow::DGCompressibleFlowOperation<dim, number>>             comp_flow_operation;
    std::unique_ptr<ObstacleField<dim, number, SphericalParticle<dim, number>>> obstacle_field;
    std::unique_ptr<Profiling::ProfilingMonitor<number>>                        profiling_monitor;
    AMR::BinaryOpIndicatorComposite<dim, number, std::plus>                     amr_indicator;

    unsigned int comp_flow_dof_idx{};
    unsigned int comp_flow_quad_idx{};

    std::unique_ptr<Postprocessor<dim, number>> post_processor;
  };

} // namespace MeltPoolDG
