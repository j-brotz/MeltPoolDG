/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Münch, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>
// MeltPoolDG
#include <meltpooldg/evaporation/evaporation_operation.hpp>
#include <meltpooldg/evaporation/evaporation_operation_marching_cube.hpp>
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/flow/flow_base.hpp>
#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/melt_pool/melt_pool_operation.hpp>
#include <meltpooldg/utilities/amr.hpp>
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
    run(std::shared_ptr<SimulationBase<dim>> base_in) final
    {
      initialize(base_in);

      while (!time_iterator.is_finished())
        {
          const auto dt = time_iterator.get_next_time_increment();
          const auto n  = time_iterator.get_current_time_step_number();

          scratch_data->get_pcout()
            << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();

          // ... solve level-set problem with the given advection field
          if (evaporation_operation)
            {
              /*
               If evaporative mass flux is considered the interface velocity will be modified.
               Note that the normal vector is used from the old step.
               */
              level_set_operation.update_normal_vector();

              if (melt_pool_operation)
                evaporation_operation->compute_evaporative_mass_flux_from_temperature(
                  melt_pool_operation->get_temperature(),
                  temp_dof_idx,
                  base_in->parameters.mp.boiling_temperature,
                  base_in->parameters.recoil.pressure_constant,
                  base_in->parameters.recoil.temperature_constant);
              else
                evaporation_operation->get_evaporative_mass_flux() =
                  base_in->parameters.evapor.evaporative_mass_flux;

              if (base_in->parameters.evapor.formulation_source_term_continuity == "diffuse")
                evaporation_operation->compute_evaporation_velocity();
#ifdef MELT_POOL_DG_WITH_ADAFLO
              else if (base_in->parameters.evapor.formulation_source_term_continuity == "sharp")
                Evaporation::EvaporationOperationMarchingCube<dim>::compute_evaporation_velocity(
                  *scratch_data,
                  evaporation_operation->get_velocity(),
                  evaporation_operation->get_evaporative_mass_flux(),
                  level_set_operation.get_level_set_as_heaviside(),
                  level_set_operation.get_normal_vector(),
                  base_in->parameters.evapor.density_liquid,
                  base_in->parameters.evapor.density_gas,
                  evapor_vel_dof_idx,
                  ls_hanging_nodes_dof_idx,
                  ls_quad_idx,
                  normal_dof_idx);
#endif
              else
                AssertThrow(false, ExcNotImplemented());
            }

          level_set_operation.solve(dt, flow_operation->get_velocity());

          // update the two phases
          update_phases(level_set_operation.get_level_set_as_heaviside(), base_in->parameters);

          if (!melt_pool_operation && heat_operation)
            {
              heat_operation->solve(dt);
            }

          // accumulate forces: a) gravity force
          compute_gravity_force(vel_force_rhs, base_in->parameters.base.gravity, true);

          // ... b) surface tension
          if (base_in->parameters.flow.temperature_dependent_surface_tension_coefficient == 0.0)
            SurfaceTensionOperation<dim>::compute_surface_tension(
              vel_force_rhs,
              *scratch_data,
              level_set_operation.get_level_set_as_heaviside(),
              level_set_operation.get_curvature(),
              base_in->parameters.flow.surface_tension_coefficient,
              ls_dof_idx,
              curv_dof_idx,
              // flow_operation->get_dof_handler_idx_velocity(),
              flow_operation->get_dof_handler_idx_hanging_nodes_velocity(),
              flow_operation->get_quad_idx_velocity(),
              false /* false means not to zero out the vorce vector */);

          // ... c) temperature-dependent surface tension
          if (!melt_pool_operation && heat_operation &&
              base_in->parameters.flow.temperature_dependent_surface_tension_coefficient > 0.0)
            Flow::SurfaceTensionOperation<dim>::compute_temperature_dependent_surface_tension(
              *scratch_data,
              vel_force_rhs,
              level_set_operation.get_level_set_as_heaviside(),
              level_set_operation.get_curvature(),
              heat_operation->get_temperature(),
              level_set_operation.get_normal_vector(),
              base_in->parameters.flow.surface_tension_coefficient,
              base_in->parameters.flow.temperature_dependent_surface_tension_coefficient,
              base_in->parameters.flow.surface_tension_reference_temperature,
              base_in->parameters.flow.surface_tension_coefficient_residual_fraction,
              ls_dof_idx,
              curv_dof_idx,
              normal_dof_idx,
              vel_dof_idx,
              flow_operation->get_quad_idx_velocity(),
              temp_dof_idx,
              false /*false means add to force vector*/);

          if (evaporation_operation)
            {
              scratch_data->initialize_dof_vector(mass_balance_rhs, pressure_dof_idx);

              if (base_in->parameters.evapor.formulation_source_term_continuity == "diffuse")
                evaporation_operation->compute_mass_balance_source_term(
                  mass_balance_rhs,
                  flow_operation->get_dof_handler_idx_pressure(),
                  flow_operation->get_quad_idx_pressure(),
                  true /* zero out force rhs */);
#ifdef MELT_POOL_DG_WITH_ADAFLO
              else if (base_in->parameters.evapor.formulation_source_term_continuity == "sharp")
                Evaporation::EvaporationOperationMarchingCube<dim>::
                  compute_mass_balance_source_term_sharp(
                    *scratch_data,
                    mass_balance_rhs,
                    evaporation_operation->get_evaporative_mass_flux(),
                    level_set_operation.get_level_set(),
                    base_in->parameters.evapor.density_liquid,
                    base_in->parameters.evapor.density_gas,
                    ls_hanging_nodes_dof_idx,
                    flow_operation->get_dof_handler_idx_pressure());
#endif
              else
                AssertThrow(false, ExcNotImplemented());
            }

          // ... solve melt pool operation
          // It is assumed that flow.density represents the density of the gas phase
          if (melt_pool_operation)
            {
              const double density_gas = base_in->parameters.flow.density;
              const double density_liquid =
                base_in->parameters.flow.density + base_in->parameters.flow.density_difference;

              melt_pool_operation->solve(vel_force_rhs,
                                         level_set_operation.get_level_set_as_heaviside(),
                                         density_gas,
                                         density_liquid,
                                         dt);
            }

          //  ... and set the resulting forces within the Navier-Stokes solver
          flow_operation->set_force_rhs(vel_force_rhs);

          if (evaporation_operation)
            flow_operation->set_mass_balance_rhs(mass_balance_rhs);

          // solver Navier-Stokes problem
          flow_operation->solve();

          scratch_data->get_pcout()
            << " |velocity| = " << std::setprecision(15)
            << VectorTools::compute_L2_norm<dim>(flow_operation->get_velocity(),
                                                 *scratch_data,
                                                 flow_operation->get_dof_handler_idx_velocity(),
                                                 flow_operation->get_quad_idx_velocity())
            << std::endl;
          scratch_data->get_pcout()
            << " |p| = " << std::setprecision(15)
            << VectorTools::compute_L2_norm<dim>(flow_operation->get_pressure(),
                                                 *scratch_data,
                                                 flow_operation->get_dof_handler_idx_pressure(),
                                                 flow_operation->get_quad_idx_pressure())
            << std::endl;
          // ... and output the results to vtk files.
          output_results(n);

          if (base_in->parameters.amr.do_amr)
            refine_mesh(base_in);
        }
    }

    std::string
    get_name() final
    {
      return "two_phase_flow";
    };

  private:
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize(std::shared_ptr<SimulationBase<dim>> base_in)
    {
      /*
       *  setup DoFHandler
       */
      dof_handler.reinit(*base_in->triangulation);
      dof_handler_evapor.reinit(*base_in->triangulation);

      /*
       *  setup scratch data
       */
      scratch_data = std::make_shared<ScratchData<dim>>(base_in->mpi_communicator,
                                                        base_in->parameters.base.verbosity_level,
                                                        /*do_matrix_free*/ true);

      /*
       *  setup mapping
       */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
      if (base_in->parameters.base.do_simplex)
        scratch_data->set_mapping(
          MappingFE<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree)));
      else
#endif
        scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));

      scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_dof_handler(dof_handler_evapor);
      scratch_data->attach_dof_handler(dof_handler);

      ls_hanging_nodes_dof_idx =
        scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
      ls_dof_idx         = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
      reinit_dof_idx     = scratch_data->attach_constraint_matrix(reinit_constraints_dirichlet);
      evapor_vel_dof_idx = scratch_data->attach_constraint_matrix(evapor_hanging_node_constraints);
      temp_dof_idx       = scratch_data->attach_constraint_matrix(temp_constraints_dirichlet);

      /*
       *  create quadrature rule
       */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
      if (base_in->parameters.base.do_simplex)
        {
          ls_quad_idx = scratch_data->attach_quadrature(
            QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
        }
      else
#endif
        {
          ls_quad_idx =
            scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
        }

#ifdef MELT_POOL_DG_WITH_ADAFLO
      flow_operation = std::make_shared<AdafloWrapper<dim>>(*scratch_data, base_in);
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *  set indices of flow dof handlers
       */
      vel_dof_idx      = flow_operation->get_dof_handler_idx_velocity();
      pressure_dof_idx = flow_operation->get_dof_handler_idx_pressure();

      setup_dof_system(base_in, false);

#ifdef MELT_POOL_DG_WITH_ADAFLO
      dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->initialize(base_in);
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *  initialize the time stepping scheme
       */
      time_iterator.initialize(
        TimeIteratorData<double>{base_in->parameters.flow.start_time,
                                 base_in->parameters.flow.end_time,
                                 base_in->parameters.flow.time_step_size,
                                 base_in->parameters.flow.max_n_steps,
                                 false /*cfl_condition-->not supported yet*/});
      /*
       *  set initial conditions of the levelset function
       */
      AssertThrow(
        base_in->get_initial_condition("level_set"),
        ExcMessage(
          "It seems that your SimulationBase object does not contain "
          "a valid initial field function for the level set field. A shared_ptr to your initial field "
          "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
          "  this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), 'level_set') "));

      /*
       *    compute intial conditions of the level set
       */
      VectorType initial_solution;
      scratch_data->initialize_dof_vector(initial_solution, ls_dof_idx);

      dealii::VectorTools::project(scratch_data->get_mapping(),
                                   dof_handler,
                                   ls_constraints_dirichlet,
                                   scratch_data->get_quadrature(ls_quad_idx),
                                   *base_in->get_initial_condition("level_set"),
                                   initial_solution);
      initial_solution.update_ghost_values();
      /*
       *    initialize the levelset operation class
       */
      level_set_operation.initialize(scratch_data,
                                     initial_solution,
                                     flow_operation->get_velocity(),
                                     base_in,
                                     ls_dof_idx,
                                     ls_hanging_nodes_dof_idx,
                                     ls_quad_idx,
                                     reinit_dof_idx,
                                     reinit_hanging_nodes_dof_idx,
                                     curv_dof_idx,
                                     normal_dof_idx,
                                     vel_dof_idx,
                                     ls_dof_idx /* todo: ls_zero_bc_idx*/);
      /*
       *    initialize the heat operation class
       */
      if (base_in->parameters.base.problem_name == "two_phase_flow_with_heat_transfer")
        {
          heat_operation = std::make_shared<Heat::HeatTransferOperation<dim>>(
            base_in->get_bc("heat_transfer"),
            *scratch_data,
            base_in->parameters.heat,
            temp_dof_idx,
            temp_hanging_nodes_dof_idx,
            temp_quad_idx,
            vel_dof_idx,
            &flow_operation->get_velocity(),
            ls_dof_idx,
            &level_set_operation.get_level_set_as_heaviside(),
            &base_in->parameters.material);

          /*
           *    compute initial conditions of the temperature field
           */
          scratch_data->initialize_dof_vector(initial_solution, temp_dof_idx);

          dealii::VectorTools::project(scratch_data->get_mapping(),
                                       dof_handler,
                                       scratch_data->get_constraint(temp_dof_idx),
                                       scratch_data->get_quadrature(temp_quad_idx),
                                       *base_in->get_initial_condition("heat_transfer"),
                                       initial_solution);
          initial_solution.update_ghost_values();

          heat_operation->set_initial_condition(initial_solution);
        }

      /*
       *    initialize the evaporation class
       */
      if (base_in->parameters.base.problem_name == "two_phase_flow_with_evaporation" ||
          base_in->parameters.base.problem_name == "melt_pool_with_evaporation")
        {
          evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim>>(
            scratch_data,
            level_set_operation.get_level_set_as_heaviside(),
            level_set_operation.get_normal_vector(),
            base_in,
            normal_dof_idx,
            evapor_vel_dof_idx,
            ls_hanging_nodes_dof_idx,
            ls_quad_idx);

          level_set_operation.setup_with_evaporation(
            temp_dof_idx,
            flow_operation->get_dof_handler_idx_hanging_nodes_velocity(),
            evapor_vel_dof_idx,
            flow_operation->get_velocity(),
            evaporation_operation->get_velocity());
        }

      /*
       *    initialize the melt pool operation class
       */
      if (base_in->parameters.base.problem_name == "melt_pool" ||
          base_in->parameters.base.problem_name == "melt_pool_with_evaporation")
        melt_pool_operation = std::make_shared<MeltPool::MeltPoolOperation<dim>>(
          scratch_data,
          base_in->get_bc("heat_transfer"),
          base_in->parameters,
          ls_dof_idx,
          reinit_dof_idx,
          flow_operation->get_dof_handler_idx_velocity(),
          flow_operation->get_quad_idx_velocity(),
          temp_dof_idx,
          temp_quad_idx,
          base_in->parameters.flow.start_time,
          evaporation_operation == nullptr);

      /*
       * set initial condition of the melt pool class
       */
      if (melt_pool_operation)
        {
          const double density_gas = base_in->parameters.flow.density;
          const double density_liquid =
            base_in->parameters.flow.density + base_in->parameters.flow.density_difference;

          melt_pool_operation->set_initial_condition(
            level_set_operation.get_level_set_as_heaviside(),
            level_set_operation.get_level_set(),
            density_gas,
            density_liquid);
        }
      /*
       *  initialize postprocessor
       */
      post_processor =
        std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(vel_dof_idx),
                                             base_in->parameters.paraview,
                                             scratch_data->get_mapping(),
                                             scratch_data->get_triangulation(vel_dof_idx));
      /*
       *  output results of initialization --> initial refinement is done afterwards (!)
       *  @todo: find a way to plot vectors on the refined mesh, which are only relevant for output
       *  and which must not be transferred to the new mesh everytime refine_mesh() is called.
       */
      output_results(0);
      /*
       *    Do initial refinement steps if requested
       */
      if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
        for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
          {
            scratch_data->get_pcout()
              << "cycle: " << i << " n_dofs: " << dof_handler.n_dofs() << "(ls) + "
              << flow_operation->get_dof_handler_velocity().n_dofs() << "(vel) + "
              << flow_operation->get_dof_handler_pressure().n_dofs() << "(p)";

            if (melt_pool_operation)
              scratch_data->get_pcout()
                << " T.size " << melt_pool_operation->get_temperature().size() << " solid.size "
                << melt_pool_operation->get_solid().size();
            if (heat_operation)
              scratch_data->get_pcout() << " T.size " << heat_operation->get_temperature().size();

            scratch_data->get_pcout() << std::endl;

            refine_mesh(base_in);
          }
    }

    void
    setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_reinit = true)
    {
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
      if (base_in->parameters.base.do_simplex)
        {
          dof_handler.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
          dof_handler_evapor.distribute_dofs(
            FESystem<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree), dim));
        }
      else
#endif
        {
          dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
          dof_handler_evapor.distribute_dofs(
            FESystem<dim>(FE_Q<dim>(base_in->parameters.base.degree), dim));
        }
        /*
         *    initialize the flow operation class
         */
#ifdef MELT_POOL_DG_WITH_ADAFLO
      dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->reinit_1();
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *  create partitioning
       */
      scratch_data->create_partitioning();
      /*
       *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
       *  dirichlet constraints are supported)
       */
      ls_hanging_node_constraints.clear();
      ls_hanging_node_constraints.reinit(
        scratch_data->get_locally_relevant_dofs(ls_hanging_nodes_dof_idx));
      DoFTools::make_hanging_node_constraints(dof_handler, ls_hanging_node_constraints);

      ls_constraints_dirichlet.clear();
      ls_constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs(ls_dof_idx));
      if (base_in->get_bc("level_set") && !base_in->get_dirichlet_bc("level_set").empty())
        {
          for (const auto &bc : base_in->get_dirichlet_bc(
                 "level_set")) // @todo: add name of bc at a more central place
            {
              dealii::VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                               dof_handler,
                                                               bc.first,
                                                               *bc.second,
                                                               ls_constraints_dirichlet);
            }
        }

      temp_constraints_dirichlet.clear();
      temp_constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs(temp_dof_idx));
      if (base_in->get_bc("heat_transfer") && !base_in->get_dirichlet_bc("heat_transfer").empty())
        {
          for (const auto &bc : base_in->get_dirichlet_bc(
                 "heat_transfer")) // @todo: add name of bc at a more central place
            {
              dealii::VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                               dof_handler,
                                                               bc.first,
                                                               *bc.second,
                                                               temp_constraints_dirichlet);
            }
        }

      reinit_constraints_dirichlet.clear();
      reinit_constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs());
      if (base_in->get_bc("reinitialization") &&
          !base_in->get_dirichlet_bc("reinitialization").empty())
        {
          for (const auto &bc : base_in->get_dirichlet_bc(
                 "reinitialization")) // @todo: add name of bc at a more central place
            {
              dealii::VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                               dof_handler,
                                                               bc.first,
                                                               *bc.second,
                                                               reinit_constraints_dirichlet);
            }
        }

      evapor_hanging_node_constraints.clear();
      evapor_hanging_node_constraints.reinit(
        scratch_data->get_locally_relevant_dofs(evapor_vel_dof_idx));
      DoFTools::make_hanging_node_constraints(dof_handler_evapor, evapor_hanging_node_constraints);

      // periodic constraints
      for (const auto &bc : base_in->get_periodic_bc())
        {
          const auto [id_in, id_out, direction] = bc;
          DoFTools::make_periodicity_constraints(
            dof_handler, id_in, id_out, direction, ls_hanging_node_constraints);
          DoFTools::make_periodicity_constraints(
            dof_handler, id_in, id_out, direction, ls_constraints_dirichlet);
          DoFTools::make_periodicity_constraints(
            dof_handler, id_in, id_out, direction, reinit_constraints_dirichlet);
          DoFTools::make_periodicity_constraints(
            dof_handler_evapor, id_in, id_out, direction, evapor_hanging_node_constraints);
          DoFTools::make_periodicity_constraints(
            dof_handler, id_in, id_out, direction, temp_constraints_dirichlet);
        }

      // finalize constraints
      ls_hanging_node_constraints.close();

      ls_constraints_dirichlet.close();
      ls_constraints_dirichlet.merge(
        ls_hanging_node_constraints,
        AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

      reinit_constraints_dirichlet.close();
      reinit_constraints_dirichlet.merge(
        ls_hanging_node_constraints,
        AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

      evapor_hanging_node_constraints.close();

      temp_constraints_dirichlet.close();
      temp_constraints_dirichlet.merge(
        ls_hanging_node_constraints,
        AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

      scratch_data->build();

      if (do_reinit)
        {
          level_set_operation.reinit();

          if (evaporation_operation)
            evaporation_operation->reinit();
          if (melt_pool_operation)
            melt_pool_operation->reinit();
          if (heat_operation)
            heat_operation->reinit();
        }

#ifdef MELT_POOL_DG_WITH_ADAFLO
      dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->reinit_2();
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *    initialize the force vector for calculating surface tension
       */
      scratch_data->initialize_dof_vector(vel_force_rhs, vel_dof_idx);
      /*
       *    initialize the force vector for calculating surface tension
       */
      if (evaporation_operation)
        {
          evaporation_operation->reinit();
          scratch_data->initialize_dof_vector(mass_balance_rhs, pressure_dof_idx);
        }
    }

    /**
     * Update material parameter of the phases.
     *
     * @todo Find a better place.
     *
     * @todo: generalize for level set value gas is 1.0
     */
    void
    update_phases(const VectorType &src, const Parameters<double> &parameters) const
    {
      double dummy;

      double mass = 0.0;

      bool variable_props =
        parameters.flow.variable_properties_over_interface == "true" ||
        parameters.flow.variable_properties_over_interface == "consistent_with_evaporation";

      scratch_data->get_matrix_free().template cell_loop<double, VectorType>(
        [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
          FECellIntegrator<dim, 1, double> ls_values(matrix_free,
                                                     ls_dof_idx,
                                                     flow_operation->get_quad_idx_velocity());

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              ls_values.reinit(cell);
              ls_values.read_dof_values_plain(src);
              ls_values.evaluate(true, false);

              for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
                {
                  const auto indicator = variable_props ?
                                           ls_values.get_value(q) :
                                           UtilityFunctions::heaviside(ls_values.get_value(q), 0.5);

                  // set density
                  if (parameters.flow.variable_properties_over_interface ==
                      "consistent_with_evaporation")
                    {
                      const double &rho_g = parameters.evapor.density_gas;
                      const double &rho_l = parameters.evapor.density_liquid;
                      flow_operation->get_density(cell, q) =
                        rho_g / (1. + (rho_g / rho_l - 1) * ls_values.get_value(q));
                    }
                  else
                    flow_operation->get_density(cell, q) =
                      parameters.flow.density + parameters.flow.density_difference * indicator;

                  // set viscosity
                  flow_operation->get_viscosity(cell, q) =
                    parameters.flow.viscosity + parameters.flow.viscosity_difference * indicator;

                  // check if no spurious densities or viscosities are computed
                  const double min_density =
                    std::min(parameters.flow.density,
                             parameters.flow.density + parameters.flow.density_difference);
                  const double max_density =
                    std::max(parameters.flow.density,
                             parameters.flow.density + parameters.flow.density_difference);

                  for (auto dens : flow_operation->get_density(cell, q))
                    if (min_density > dens || dens > max_density)
                      std::cout << "WARNING: density does not comply with input:" << dens
                                << std::endl;

                  const double min_viscosity =
                    std::min(parameters.flow.viscosity,
                             parameters.flow.viscosity + parameters.flow.viscosity_difference);
                  const double max_viscosity =
                    std::max(parameters.flow.viscosity,
                             parameters.flow.viscosity + parameters.flow.viscosity_difference);

                  for (auto visc : flow_operation->get_viscosity(cell, q))
                    if (min_viscosity > visc || visc > max_viscosity)
                      std::cout << "WARNING: viscosity does not comply with input:" << visc
                                << std::endl;

                  // compute overall mass
                  for (unsigned int v = 0;
                       v < scratch_data->get_matrix_free().n_active_entries_per_cell_batch(cell);
                       ++v)
                    {
                      mass += (parameters.flow.density +
                               parameters.flow.density_difference * ls_values.get_value(q)[v]) *
                              ls_values.JxW(q)[v];
                    }
                }
            }
        },
        dummy,
        src);

      if (evaporation_operation || melt_pool_operation)
        {
          scratch_data->get_pcout()
            << "    | two phase flow: total mass = "
            << Utilities::MPI::sum(mass, scratch_data->get_mpi_comm()) << std::endl;
        }
    }

    /**
     * Compute gravity force.
     *
     * @todo Find a better place.
     */
    void
    compute_gravity_force(VectorType &vec, const double gravity, const bool zero_out = true) const
    {
      scratch_data->get_matrix_free().template cell_loop<VectorType, std::nullptr_t>(
        [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
          FECellIntegrator<dim, dim, double> force_values(matrix_free,
                                                          vel_dof_idx,
                                                          flow_operation->get_quad_idx_velocity());

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              force_values.reinit(cell);

              for (unsigned int q = 0; q < force_values.n_q_points; ++q)
                {
                  Tensor<1, dim, VectorizedArray<double>> force;

                  force[dim - 1] -= gravity * flow_operation->get_density(cell, q);
                  force_values.submit_value(force, q);
                }
              force_values.integrate_scatter(true, false, vec);
            }
        },
        vec,
        nullptr,
        zero_out);
    }

    /*
     *  This function is to create paraview output
     */
    void
    output_results(const unsigned int n_time_step)
    {
      /**
       * collect all relevant output data
       */
      const auto attach_output_vectors = [&](DataOut<dim> &data_out) {
        level_set_operation.attach_output_vectors(data_out);

        flow_operation->attach_output_vectors(data_out);

        if (melt_pool_operation)
          melt_pool_operation->attach_output_vectors(data_out);

        if (evaporation_operation)
          evaporation_operation->attach_output_vectors(data_out);
        if (heat_operation)
          heat_operation->attach_output_vectors(data_out);
      };
      /**
       * do the output operation
       */
      post_processor->process(n_time_step, attach_output_vectors);
    }
    /*
     *  perform mesh refinement
     */
    void
    refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
    {
      const auto mark_cells_for_refinement =
        [&](parallel::distributed::Triangulation<dim> &tria) -> bool {
        Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

        VectorType locally_relevant_solution;
        locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));

        locally_relevant_solution.copy_locally_owned_data_from(level_set_operation.get_level_set());
        ls_constraints_dirichlet.distribute(locally_relevant_solution);
        locally_relevant_solution.update_ghost_values();

        for (unsigned int i = 0; i < locally_relevant_solution.local_size(); ++i)
          locally_relevant_solution.local_element(i) =
            (1.0 - locally_relevant_solution.local_element(i) *
                     locally_relevant_solution.local_element(i));

        locally_relevant_solution.update_ghost_values();

        dealii::VectorTools::integrate_difference(scratch_data->get_dof_handler(ls_dof_idx),
                                                  locally_relevant_solution,
                                                  Functions::ZeroFunction<dim>(),
                                                  estimated_error_per_cell,
                                                  scratch_data->get_quadrature(ls_quad_idx),
                                                  dealii::VectorTools::L2_norm);

        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
          tria,
          estimated_error_per_cell,
          base_in->parameters.amr.upper_perc_to_refine,
          base_in->parameters.amr.lower_perc_to_coarsen);

        return true;
      };

      std::vector<
        std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
        data;

      data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
        level_set_operation.attach_vectors(vectors); // ls + heaviside
      });
      data.emplace_back(&flow_operation->get_dof_handler_velocity(),
                        [&](std::vector<VectorType *> &vectors) {
                          flow_operation->attach_vectors_u(vectors);
                        });
      data.emplace_back(&flow_operation->get_dof_handler_pressure(),
                        [&](std::vector<VectorType *> &vectors) {
                          flow_operation->attach_vectors_p(vectors);
                        });

      if (melt_pool_operation)
        data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
          melt_pool_operation->attach_vectors(vectors); // temperature + solid + liquid
        });

      if (evaporation_operation)
        {
          data.emplace_back(&dof_handler_evapor, [&](std::vector<VectorType *> &vectors) {
            evaporation_operation->attach_dim_vectors(vectors);
          });
          data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
            evaporation_operation->attach_vectors(vectors);
          });
        }

      if (heat_operation)
        data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
          heat_operation->attach_vectors(vectors);
        });

      const auto post = [&]() {
        /**
         * level set
         */
        ls_constraints_dirichlet.distribute(level_set_operation.get_level_set());
        ls_hanging_node_constraints.distribute(level_set_operation.get_level_set_as_heaviside());

        /**
         * flow
         */
        flow_operation->distribute_constraints();

        /**
         * melt pool
         */
        if (melt_pool_operation)
          melt_pool_operation->distribute_constraints();
        /**
         * evaporation
         */
        if (evaporation_operation)
          evaporation_operation->distribute_constraints();
        /**
         * melt pool
         */
        if (heat_operation)
          heat_operation->distribute_constraints();
      };

      const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

      refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                   data,
                                   post,
                                   setup_dof_system,
                                   base_in->parameters.amr,
                                   time_iterator.get_current_time_step_number());
    }

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
