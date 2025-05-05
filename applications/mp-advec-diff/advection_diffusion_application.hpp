#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation_base.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

#include "advection_diffusion_case.hpp"

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class AdvectionDiffusionApplication
  {
  private:
    using CaseType        = AdvectionDiffusionCase<dim, number>;
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    AdvectionDiffusionApplication(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();

  private:
    std::unique_ptr<CaseType> simulation_case;

    dealii::DoFHandler<dim>                        dof_handler;
    dealii::AffineConstraints<number>              constraints;
    dealii::AffineConstraints<number>              hanging_node_constraints;
    dealii::AffineConstraints<number>              hanging_node_constraints_with_zero_dirichlet;
    dealii::DoFHandler<dim>                        dof_handler_velocity;
    dealii::AffineConstraints<number>              hanging_node_constraints_velocity;
    std::shared_ptr<ScratchData<dim, dim, number>> scratch_data;
    VectorType                                     advection_velocity;
    std::unique_ptr<TimeIntegration::TimeIterator<number>>        time_iterator;
    std::unique_ptr<AdvectionDiffusionOperationBase<dim, number>> advec_diff_operation;
    std::unique_ptr<Profiling::ProfilingMonitor<number>>          profiling_monitor;


    unsigned int advec_diff_dof_idx;
    unsigned int advec_diff_hanging_nodes_dof_idx;
    unsigned int advec_diff_adaflo_dof_idx;
    unsigned int velocity_dof_idx;

    unsigned int advec_diff_quad_idx;

    std::unique_ptr<Postprocessor<dim, number>> post_processor;

    void
    setup_dof_system();

    void
    initialize();

    void
    compute_advection_velocity(dealii::Function<dim, number> &advec_func);

    void
    output_results(unsigned int time_step, number current_time);

    void
    refine_mesh();
  };

} // namespace MeltPoolDG::LevelSet
