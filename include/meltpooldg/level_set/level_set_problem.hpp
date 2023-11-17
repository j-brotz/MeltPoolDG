/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/evaporation/evaporation_operation.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  BETTER_ENUM(AMRStrategy, char, generic, refine_all_interface_cells)

  template <int dim>
  class LevelSetProblem : public ProblemBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    std::shared_ptr<ScratchData<dim>> scratch_data;

    std::shared_ptr<TimeIterator<double>> time_iterator;

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

    std::shared_ptr<LevelSetOperation<dim>>                 level_set_operation;
    std::shared_ptr<Evaporation::EvaporationOperation<dim>> evaporation_operation;

    VectorType advection_velocity;
    VectorType initial_solution;

    std::shared_ptr<Postprocessor<dim>> post_processor;

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
    initialize(std::shared_ptr<SimulationBase<dim>> base_in);

    void
    setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_reinit = true);

    void
    compute_advection_velocity(Function<dim> &advec_func);
    /*
     *  perform output of results
     */
    void
    output_results(const unsigned int                   time_step,
                   const double                         current_time,
                   std::shared_ptr<SimulationBase<dim>> base_in);
    /*
     *  perform mesh refinement
     */
    void
    refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in);

  protected:
    void
    add_parameters(dealii::ParameterHandler &) final;

  public:
    LevelSetProblem() = default;

    void
    run(std::shared_ptr<SimulationBase<dim>> base_in) final;

    std::string
    get_name() final;
  };
} // namespace MeltPoolDG::LevelSet
