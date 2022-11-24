/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Munch, TUM, October 2020
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

#include <deal.II/numerics/error_estimator.h>
// MeltPoolDG
#include <meltpooldg/evaporation/evaporation_operation.hpp>
#include <meltpooldg/evaporation/incompressible_newtonian_evaporation_material.hpp>
#include <meltpooldg/flow/darcy_damping_operation.hpp>
#include <meltpooldg/flow/flow_base.hpp>
#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/interface/problem_base.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/melt_pool/melt_pool_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/restart.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::MeltPool
{
  using namespace dealii;

  BETTER_ENUM(AMRStrategy, char, generic, adaflo, KellyErrorEstimator)
  BETTER_ENUM(AutomaticGridRefinementType, char, fixed_fraction, fixed_number)


  template <int dim>
  class MeltPoolProblem : public ProblemBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    MeltPoolProblem() = default;

    void
    run(std::shared_ptr<SimulationBase<dim>> base_in) final;

    std::string
    get_name() final;

  protected:
    void
    add_parameters(dealii::ParameterHandler &) final;

    void
    check_input_parameters(Parameters<double> &) final;

  private:
    void
    save(std::shared_ptr<SimulationBase<dim>> base_in);

    void
    load(std::shared_ptr<SimulationBase<dim>> base_in);

    struct
    {
      bool do_heat_transfer = false;
      bool do_evaporative_heat_flux =
        false; //@todo: move to struct "phase change", rename do_evaporative_enthalpy_jump
      bool do_evaporative_velocity_jump  = false; //@todo: move to struct "phase change"
      bool do_recoil_pressure            = false; //@todo: move to struct "phase change", rename
      bool do_melt_pool                  = false;
      bool do_advect_level_set           = true;
      bool do_extrapolate_coupling_terms = false;

      struct
      {
        // number of iterations to balance nonlinearity in advection diffusion equation with
        // evaporation
        int    n_max_iter = 1;
        double tol        = 1e-10;
      } level_set_evapor_coupling;

      struct
      {
        AMRStrategy                 strategy = AMRStrategy::generic;
        AutomaticGridRefinementType automatic_grid_refinement_type =
          AutomaticGridRefinementType::fixed_number;
        bool   do_auto_detect_frequency                   = false;
        bool   do_refine_all_interface_cells              = false;
        double fraction_of_melting_point_refined_in_solid = 1.0;
      } amr;

      struct
      {
        double time_step_size                   = -1;
        double max_temperature                  = -1;
        double max_change_factor_time_step_size = 1.5;
      } mp_heat_up;

    } problem_specific_parameters;

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
     */
    void
    update_phases(const VectorType &ls_as_heaviside, const Parameters<double> &parameters) const;

    /**
     * Compute gravity force.
     *
     * @todo Move to own class.
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

    /*
     *  perform mesh refinement
     */
    void
    attach_vectors(std::vector<std::pair<const DoFHandler<dim> *,
                                         std::function<void(std::vector<VectorType *> &)>>> &data);

    /*
     *  perform mesh refinement
     */
    void
    post();

    bool
    mark_cells_for_refinement(std::shared_ptr<SimulationBase<dim>> base_in,
                              Triangulation<dim> &                 tria);

    std::shared_ptr<TimeIterator<double>> time_iterator;

    DoFHandler<dim> dof_handler_ls;
    DoFHandler<dim> dof_handler_heat;

    AffineConstraints<double> ls_constraints_dirichlet;
    AffineConstraints<double> ls_hanging_node_constraints;
    AffineConstraints<double> reinit_constraints_dirichlet;
    AffineConstraints<double> reinit_no_solid_constraints_dirichlet;
    AffineConstraints<double> temp_constraints_dirichlet;
    AffineConstraints<double> temp_hanging_node_constraints;
    AffineConstraints<double> flow_velocity_constraints_no_solid;

    VectorType vel_force_rhs;
    VectorType mass_balance_rhs;
    VectorType level_set_rhs;
    VectorType interface_velocity;

    unsigned int ls_dof_idx;
    unsigned int ls_hanging_nodes_dof_idx;
    unsigned int ls_quad_idx;
    unsigned int reinit_dof_idx;
    unsigned int reinit_no_solid_dof_idx;
    unsigned int temp_dof_idx;
    unsigned int vel_dof_idx;
    unsigned int pressure_dof_idx;
    unsigned int flow_vel_no_solid_dof_idx;
    unsigned int temp_hanging_nodes_dof_idx;
    unsigned int temp_quad_idx;

    const unsigned int &curv_dof_idx             = ls_hanging_nodes_dof_idx;
    const unsigned int &normal_dof_idx           = ls_hanging_nodes_dof_idx;
    const unsigned int &evapor_vel_dof_idx       = vel_dof_idx;
    const unsigned int &evapor_mass_flux_dof_idx = temp_hanging_nodes_dof_idx;

    std::shared_ptr<ScratchData<dim>>                       scratch_data;
    std::shared_ptr<Material<double>>                       material;
    std::shared_ptr<Flow::FlowBase<dim>>                    flow_operation;
    std::shared_ptr<LevelSet::LevelSetOperation<dim>>       level_set_operation;
    std::shared_ptr<MeltPool::MeltPoolOperation<dim>>       melt_pool_operation;
    std::shared_ptr<Evaporation::EvaporationOperation<dim>> evaporation_operation = nullptr;
    std::shared_ptr<
      MeltPoolDG::Evaporation::IncompressibleNewtonianFluidEvaporationMaterial<dim, double>>
                                                        evaporation_fluid_material;
    std::shared_ptr<Heat::HeatTransferOperation<dim>>   heat_operation;
    std::shared_ptr<Flow::DarcyDampingOperation<dim>>   darcy_operation;
    std::shared_ptr<Flow::SurfaceTensionOperation<dim>> surface_tension_operation;
    std::shared_ptr<Postprocessor<dim>>                 post_processor;
    std::shared_ptr<Restart::RestartMonitor<double>>    restart_monitor;
  };
} // namespace MeltPoolDG::MeltPool
