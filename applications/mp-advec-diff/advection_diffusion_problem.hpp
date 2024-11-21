/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_operation_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include "advection_diffusion_case.hpp"

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;
  template <int dim>
  class AdvectionDiffusionProblem
  {
  private:
    using CaseType        = AdvectionDiffusionCase<dim>;
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    AdvectionDiffusionProblem(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();

  private:
    /*
     *  This function initials the relevant member data
     *  for the computation of the advection-diffusion problem
     */
    void
    setup_dof_system();

    void
    initialize();

    void
    compute_advection_velocity(Function<dim> &advec_func);

    /*
     *  perform output of results
     */
    void
    output_results(unsigned int time_step, double current_time);

    /*
     *  perform mesh refinement
     */
    void
    refine_mesh();

    std::unique_ptr<CaseType> simulation_case;

    DoFHandler<dim>                       dof_handler;
    AffineConstraints<double>             constraints;
    AffineConstraints<double>             hanging_node_constraints;
    AffineConstraints<double>             hanging_node_constraints_with_zero_dirichlet;
    DoFHandler<dim>                       dof_handler_velocity;
    AffineConstraints<double>             hanging_node_constraints_velocity;
    std::shared_ptr<ScratchData<dim>>     scratch_data;
    VectorType                            advection_velocity;
    std::unique_ptr<TimeIterator<double>> time_iterator;
    std::unique_ptr<AdvectionDiffusionOperationBase<dim>> advec_diff_operation;
    std::unique_ptr<Profiling::ProfilingMonitor<double>>  profiling_monitor;


    unsigned int advec_diff_dof_idx;
    unsigned int advec_diff_hanging_nodes_dof_idx;
    unsigned int advec_diff_adaflo_dof_idx;
    unsigned int velocity_dof_idx;

    unsigned int advec_diff_quad_idx;

    std::unique_ptr<Postprocessor<dim>> post_processor;
  };

} // namespace MeltPoolDG::LevelSet
