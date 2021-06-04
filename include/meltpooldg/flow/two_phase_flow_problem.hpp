/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Münch, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>
// MeltPoolDG
#include <meltpooldg/evaporation/evaporation_operation.hpp>
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/flow/flow_base.hpp>
#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/melt_pool/melt_pool_operation.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/postprocessor.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim>
  class TwoPhaseFlowProblem : public ProblemBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    TwoPhaseFlowProblem() = default;

    void
    run(std::shared_ptr<SimulationBase<dim>> base_in) final;

    std::string
    get_name() final;

  private:
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize(std::shared_ptr<SimulationBase<dim>> base_in);

    void
    set_initial_condition(std::shared_ptr<SimulationBase<dim>> base_in);

    void
    setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_reinit = true);

    /**
     * Update material parameter of the phases.
     *
     * @todo Find a better place.
     *
     * @todo: generalize for level set value gas is 1.0
     */
    void
    update_phases(const VectorType &src, const Parameters<double> &parameters) const;

    /**
     * Compute gravity force.
     *
     * @todo Find a better place.
     */
    void
    compute_gravity_force(VectorType &vec, const double gravity, const bool zero_out = true) const;

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

    TimeIterator<double> time_iterator;
    DoFHandler<dim>      dof_handler;
    DoFHandler<dim>      dof_handler_evapor;

    AffineConstraints<double> ls_constraints_dirichlet;
    AffineConstraints<double> ls_hanging_node_constraints;
    AffineConstraints<double> reinit_constraints_dirichlet;
    AffineConstraints<double> evapor_hanging_node_constraints;
    AffineConstraints<double> temp_constraints_dirichlet;

    VectorType vel_force_rhs;
    VectorType mass_balance_rhs;

    unsigned int ls_dof_idx;
    unsigned int ls_hanging_nodes_dof_idx;
    unsigned int ls_quad_idx;
    unsigned int reinit_dof_idx;
    unsigned int evapor_vel_dof_idx;
    unsigned int temp_dof_idx;

    const unsigned int &reinit_hanging_nodes_dof_idx = ls_hanging_nodes_dof_idx;
    const unsigned int &curv_dof_idx                 = ls_hanging_nodes_dof_idx;
    const unsigned int &normal_dof_idx               = ls_hanging_nodes_dof_idx;
    const unsigned int &temp_quad_idx                = ls_quad_idx;
    const unsigned int &temp_hanging_nodes_dof_idx   = ls_hanging_nodes_dof_idx;

    unsigned int vel_dof_idx;
    unsigned int pressure_dof_idx;

    std::shared_ptr<ScratchData<dim>>                       scratch_data;
    std::shared_ptr<FlowBase<dim>>                          flow_operation;
    LevelSet::LevelSetOperation<dim>                        level_set_operation;
    std::shared_ptr<MeltPool::MeltPoolOperation<dim>>       melt_pool_operation;
    std::shared_ptr<Evaporation::EvaporationOperation<dim>> evaporation_operation = nullptr;
    std::shared_ptr<Heat::HeatTransferOperation<dim>>       heat_operation;
    std::shared_ptr<Postprocessor<dim>>                     post_processor;
  };
} // namespace MeltPoolDG::Flow
