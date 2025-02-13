#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_DG_operation.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/phase_change/evaporation_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include "level_set_case.hpp"

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  BETTER_ENUM(AMRStrategy, char, generic, refine_all_interface_cells)

  template <int dim>
  class LevelSetProblem
  {
  private:
    using CaseType        = LevelSetCase<dim>;
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const std::unique_ptr<CaseType> simulation_case;

    std::shared_ptr<ScratchData<dim>> scratch_data;

    std::unique_ptr<TimeIterator<double>> time_iterator;

    DoFHandler<dim> dof_handler;
    DoFHandler<dim> dof_handler_velocity;

    AffineConstraints<double> constraints_dirichlet;
    AffineConstraints<double> hanging_node_constraints;
    AffineConstraints<double> hanging_node_constraints_velocity;
    AffineConstraints<double> hanging_node_constraints_with_zero_dirichlet;

    unsigned int        ls_dof_idx;
    unsigned int        ls_quad_idx;
    unsigned int        ls_zero_bc_idx;
    unsigned int        ls_hanging_nodes_dof_idx;
    unsigned int        vel_dof_idx;
    const unsigned int &curv_dof_idx   = ls_hanging_nodes_dof_idx;
    const unsigned int &normal_dof_idx = ls_hanging_nodes_dof_idx;
    const unsigned int &reinit_dof_idx =
      ls_hanging_nodes_dof_idx; //@todo: would it make sense to use ls_zero_bc_idx?
    const unsigned int &reinit_hanging_nodes_dof_idx =
      ls_hanging_nodes_dof_idx; //@todo: would it make sense to use ls_zero_bc_idx?

    std::unique_ptr<LevelSetOperationBase<dim>>             level_set_operation;
    std::unique_ptr<Evaporation::EvaporationOperation<dim>> evaporation_operation;

    VectorType advection_velocity;
    VectorType initial_solution;

    std::unique_ptr<Postprocessor<dim>> post_processor;

    std::unique_ptr<Profiling::ProfilingMonitor<double>> profiling_monitor;

    struct
    {
      struct
      {
        AMRStrategy strategy = AMRStrategy::generic;
      } amr;
    } problem_specific_parameters;
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize();

    void
    setup_dof_system(const bool do_reinit = true);

    void
    compute_advection_velocity(Function<dim> &advec_func);
    /*
     *  perform output of results
     */
    void
    output_results(const unsigned int time_step, const double current_time);
    /*
     *  perform mesh refinement
     */
    void
    refine_mesh();

  public:
    LevelSetProblem(std::unique_ptr<CaseType> simulation_case)
      : simulation_case(std::move(simulation_case))
    {}

    void
    run();
  };
} // namespace MeltPoolDG::LevelSet
