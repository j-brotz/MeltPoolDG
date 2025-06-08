#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_DG_operation.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/phase_change/evaporation_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>

#include "level_set_case.hpp"

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class LevelSetApplication
  {
  private:
    using CaseType        = LevelSetCase<dim, number>;
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const std::unique_ptr<CaseType> simulation_case;

    dealii::DoFHandler<dim> dof_handler;
    dealii::DoFHandler<dim> dof_handler_velocity;

    dealii::AffineConstraints<number> constraints_dirichlet;
    dealii::AffineConstraints<number> hanging_node_constraints;
    dealii::AffineConstraints<number> hanging_node_constraints_velocity;
    dealii::AffineConstraints<number> hanging_node_constraints_with_zero_dirichlet;
    dealii::AffineConstraints<number> normal_dirichlet_x_constraints;
    dealii::AffineConstraints<number> normal_dirichlet_y_constraints;
    dealii::AffineConstraints<number> normal_dirichlet_z_constraints;

    unsigned int        ls_dof_idx;
    unsigned int        ls_quad_idx;
    unsigned int        ls_zero_bc_idx;
    unsigned int        ls_hanging_nodes_dof_idx;
    unsigned int        vel_dof_idx;
    const unsigned int &curv_dof_idx               = ls_hanging_nodes_dof_idx;
    unsigned int        normal_no_bc_dof_idx       = -1;
    unsigned int        normal_dirichlet_x_dof_idx = -1;
    unsigned int        normal_dirichlet_y_dof_idx = -1;
    unsigned int        normal_dirichlet_z_dof_idx = -1;
    const unsigned int &reinit_dof_idx =
      ls_hanging_nodes_dof_idx; //@todo: would it make sense to use ls_zero_bc_idx?
    const unsigned int &reinit_hanging_nodes_dof_idx =
      ls_hanging_nodes_dof_idx; //@todo: would it make sense to use ls_zero_bc_idx?

    VectorType advection_velocity;

    std::unique_ptr<LevelSetOperationBase<dim, number>>             level_set_operation;
    std::unique_ptr<Evaporation::EvaporationOperation<dim, number>> evaporation_operation;
    std::shared_ptr<ScratchData<dim, dim, number>>                  scratch_data;
    std::unique_ptr<TimeIntegration::TimeIterator<number>>          time_iterator;
    std::unique_ptr<Postprocessor<dim, number>>                     post_processor;
    std::unique_ptr<Profiling::ProfilingMonitor<number>>            profiling_monitor;

    void
    initialize();

    void
    setup_dof_system(const bool do_reinit = true);

    void
    compute_advection_velocity(dealii::Function<dim> &advec_func);

    void
    output_results(const unsigned int time_step, const number current_time);

    void
    refine_mesh();

  public:
    LevelSetApplication(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();
  };
} // namespace MeltPoolDG::LevelSet
