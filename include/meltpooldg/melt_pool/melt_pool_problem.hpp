#pragma once
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/problem_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/flow/darcy_damping_operation.hpp>
#include <meltpooldg/flow/flow_base.hpp>
#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/heat/heat_operation_base.hpp>
#include <meltpooldg/heat/laser_operation.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/phase_change/evaporation_operation.hpp>
#include <meltpooldg/phase_change/incompressible_newtonian_evaporation_material.hpp>
#include <meltpooldg/phase_change/melt_front_propagation.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/material.hpp>
#include <meltpooldg/utilities/profiling_monitor.hpp>
#include <meltpooldg/utilities/restart.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG::MeltPool
{
  BETTER_ENUM(AMRStrategy, char, generic, adaflo, KellyErrorEstimator)
  BETTER_ENUM(AutomaticGridRefinementType, char, fixed_fraction, fixed_number)
  BETTER_ENUM(OutputNotConvergedOperation, char, none, navier_stokes, heat_transfer)


  template <int dim, typename number>
  class MeltPoolProblem : public ProblemBase<dim, number>
  {
  private:
    using SimulationType  = MeltPoolCase<dim, number>;
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    MeltPoolProblem() = default;

    void
    run(std::shared_ptr<SimulationType> base_in) final;

  protected:
    void
    add_parameters(dealii::ParameterHandler &) final;

    void
    check_input_parameters();

  private:
    void
    save(std::shared_ptr<SimulationType> base_in);

    void
    load(std::shared_ptr<SimulationType> base_in);

    struct
    {
      bool do_heat_transfer              = false;
      bool do_solidification             = false;
      bool do_advect_level_set           = true;
      bool do_extrapolate_coupling_terms = false;

      struct
      {
        // number of iterations to balance nonlinearity in advection diffusion equation with
        // evaporation
        int    n_max_iter = 1;
        number tol        = 1e-10;
      } level_set_evapor_coupling;

      struct
      {
        // number of iterations to balance nonlinearity in heat equation with
        // evaporation
        int    n_max_iter = 1;
        number tol        = 1e-10;
      } heat_evapor_coupling;

      struct
      {
        AMRStrategy                 strategy = AMRStrategy::generic;
        AutomaticGridRefinementType automatic_grid_refinement_type =
          AutomaticGridRefinementType::fixed_number;
        bool   do_auto_detect_frequency                   = false;
        bool   do_refine_all_interface_cells              = false;
        number fraction_of_melting_point_refined_in_solid = 1.0;
        bool   refine_gas_domain                          = false;
      } amr;

      struct
      {
        number time_step_size                   = -1;
        number max_temperature                  = -1;
        number max_change_factor_time_step_size = 1.5;
      } mp_heat_up;

    } problem_specific_parameters;

    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize(std::shared_ptr<SimulationType> base_in);

    void
    set_initial_conditions(std::shared_ptr<SimulationType> base_in);

    void
    set_initial_condition_level_set(std::shared_ptr<SimulationType> base_in);

    void
    set_initial_condition_heat_transfer(std::shared_ptr<SimulationType> base_in);

    void
    set_initial_condition_flow(std::shared_ptr<SimulationType> base_in);

    void
    set_initial_condition_evaporation(std::shared_ptr<SimulationType> base_in);

    void
    setup_dof_system(std::shared_ptr<SimulationType> base_in, const bool do_reinit = true);

    /**
     * Update material parameter of the phases.
     */
    void
    set_phase_dependent_parameters_flow(const Parameters<number> &parameters);

    /**
     * Compute gravity force.
     *
     * @todo Move to own class.
     */
    void
    compute_gravity_force(VectorType &vec, const number gravity, const bool zero_out = true) const;

    void
    compute_interface_velocity(const LevelSet::LevelSetData<number>       &ls_data,
                               const Evaporation::EvaporationData<number> &evapor_data);

    void
    compute_interface_velocity_sharp(const LevelSet::LevelSetData<number>       &ls_data,
                                     const Evaporation::EvaporationData<number> &evapor_data);

    /*
     *  perform output of results
     */
    void
    output_results(std::shared_ptr<SimulationType>   base_in,
                   const bool                        force_output = false,
                   const OutputNotConvergedOperation output_not_converged_operation =
                     OutputNotConvergedOperation::none);
    /*
     * collect all relevant output data
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const;
    /*
     * finalize simulation in the case that an operaion didn't converge
     */
    void
    finalize(std::shared_ptr<SimulationType>   base_in,
             const OutputNotConvergedOperation output_no_converged_operation =
               OutputNotConvergedOperation::none);
    /*
     *  perform mesh refinement
     */
    void
    refine_mesh(std::shared_ptr<SimulationType> base_in);

    /*
     *  perform mesh refinement
     */
    void
    attach_vectors(std::vector<std::pair<const dealii::DoFHandler<dim> *,
                                         std::function<void(std::vector<VectorType *> &)>>> &data);

    /*
     *  perform mesh refinement
     */
    void
    post();

    bool
    mark_cells_for_refinement(std::shared_ptr<SimulationType> base_in,
                              dealii::Triangulation<dim>     &tria);

    std::shared_ptr<TimeIterator<number>> time_iterator;

    dealii::DoFHandler<dim> dof_handler_ls;

    // optional heat DoFHandler
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_heat;
    // optional DoFHandler for the HeatCutOperation's continuous DoFs
    std::unique_ptr<dealii::DoFHandler<dim>> dof_handler_heat_cont;

    dealii::AffineConstraints<number> ls_constraints_dirichlet;
    dealii::AffineConstraints<number> ls_hanging_node_constraints;
    dealii::AffineConstraints<number> reinit_constraints_dirichlet;
    dealii::AffineConstraints<number> reinit_no_solid_constraints_dirichlet;

    std::unique_ptr<dealii::AffineConstraints<number>> heat_dirichlet_constraints;
    std::unique_ptr<dealii::AffineConstraints<number>> heat_hanging_node_constraints;
    std::unique_ptr<dealii::AffineConstraints<number>> heat_continuous_hanging_node_constraints;

    dealii::AffineConstraints<number> flow_velocity_constraints_no_solid;

    VectorType vel_force_rhs;
    VectorType mass_balance_rhs;
    VectorType level_set_rhs;
    VectorType interface_velocity;

    unsigned int ls_dof_idx;
    unsigned int ls_hanging_nodes_dof_idx;
    unsigned int ls_quad_idx;
    unsigned int reinit_dof_idx;
    unsigned int reinit_no_solid_dof_idx;

    // optional DoFHandler indices
    // default value is invalid so we don't accidentally use a different DoFHandler
    unsigned int heat_dof_idx                  = -1;
    unsigned int heat_no_bc_dof_idx            = -1;
    unsigned int heat_continuous_no_bc_dof_idx = -1;
    unsigned int heat_quad_idx                 = -1;

    unsigned int vel_dof_idx;
    unsigned int pressure_dof_idx;
    unsigned int flow_vel_no_solid_dof_idx;

    const unsigned int &curv_dof_idx             = ls_hanging_nodes_dof_idx;
    const unsigned int &normal_dof_idx           = ls_hanging_nodes_dof_idx;
    const unsigned int &evapor_vel_dof_idx       = vel_dof_idx;
    const unsigned int &evapor_mass_flux_dof_idx = heat_no_bc_dof_idx;

    std::shared_ptr<ScratchData<dim, dim, number>>                  scratch_data;
    std::shared_ptr<Material<number>>                               material;
    std::shared_ptr<Flow::FlowBase<dim>>                            flow_operation;
    std::shared_ptr<LevelSet::LevelSetOperation<dim, number>>       level_set_operation;
    std::shared_ptr<Heat::LaserOperation<dim, number>>              laser_operation;
    std::shared_ptr<MeltFrontPropagation<dim, number>>              melt_front_propagation;
    std::shared_ptr<Evaporation::EvaporationOperation<dim, number>> evaporation_operation = nullptr;
    std::shared_ptr<Evaporation::IncompressibleNewtonianFluidEvaporationMaterial<dim, number>>
                                                                       evaporation_fluid_material;
    std::shared_ptr<Heat::HeatOperationBase<dim, number>>              heat_operation;
    std::shared_ptr<Flow::DarcyDampingOperation<dim, number>>          darcy_operation;
    std::shared_ptr<Flow::SurfaceTensionOperation<dim, number>>        surface_tension_operation;
    std::shared_ptr<Evaporation::RecoilPressureOperation<dim, number>> recoil_pressure_operation;
    std::shared_ptr<Postprocessor<dim, number>>                        post_processor;
    std::unique_ptr<Profiling::ProfilingMonitor<number>>               profiling_monitor;
    std::shared_ptr<Restart::RestartMonitor<number>>                   restart_monitor;

    bool output_interface_velocity     = false;
    bool compute_interface_temperature = false;
  };
} // namespace MeltPoolDG::MeltPool
