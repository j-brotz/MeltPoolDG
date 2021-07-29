#ifndef MELT_POOL_DG_DIM
#  define MELT_POOL_DG_DIM 1
#endif

#include <meltpooldg/melt_pool/melt_pool_problem.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim>
  void
  MeltPoolProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator.is_finished())
      {
        const auto dt = time_iterator.get_next_time_increment();
        const auto n  = time_iterator.get_current_time_step_number();

        time_iterator.print_me(scratch_data->get_pcout());

        /******************************************************************************************
         * LEVEL SET
         ******************************************************************************************/
        interface_velocity.copy_locally_owned_data_from(flow_operation->get_velocity());

        if (evaporation_operation)
          {
            /*
             * If evaporative mass flux is considered the interface velocity will be modified.
             * Note that the normal vector is used from the old step.
             */
            level_set_operation.update_normal_vector();

            /*
             * compute the evaporative mass flux
             */
            evaporation_operation->compute_evaporative_mass_flux();
            /*
             * compute level set source term from evaporation
             */
            evaporation_operation->compute_evaporation_velocity();

            interface_velocity += evaporation_operation->get_velocity();
          }

        // ... solve level-set problem with the given advection field
        level_set_operation.solve(dt, interface_velocity);

        /******************************************************************************************
         * HEAT TRANSFER
         ******************************************************************************************/
        if (melt_pool_operation)
          melt_pool_operation->compute_heat_source(heat_operation->get_heat_source(),
                                                   level_set_operation.get_level_set_as_heaviside(),
                                                   level_set_operation.get_normal_vector(),
                                                   normal_dof_idx,
                                                   dt,
                                                   true /* zero_out */);

        // the heat equation will NOT be solved if
        //    * the evaporative mass flux is given as a constant (temperature-independent) value
        //    * the temperature field is prescribed analytically
        if ((heat_operation && !evaporation_operation && !melt_pool_operation) ||
            (evaporation_operation &&
             !(base_in->parameters.evapor.evaporation_model == "constant")) ||
            (melt_pool_operation && !(base_in->parameters.laser.heat_source_model == "Analytical")))
          heat_operation->solve(dt);

        if (melt_pool_operation)
          melt_pool_operation->compute_melt_front_propagation(
            level_set_operation.get_level_set_as_heaviside());

        /******************************************************************************************
         * NAVIER - STOKES
         ******************************************************************************************/

        // update the phases for the flow solver considering the updated level set and temperature
        update_phases(level_set_operation.get_level_set_as_heaviside(), base_in->parameters);

        // ... a) gravity force
        compute_gravity_force(vel_force_rhs,
                              base_in->parameters.base.gravity,
                              true /* true means force vector is zeroed out before */);

        // ... b) (temperature-independent) surface tension
        if (base_in->parameters.flow.temperature_dependent_surface_tension_coefficient == 0.0)
          SurfaceTensionOperation<dim>::compute_surface_tension(
            vel_force_rhs,
            *scratch_data,
            level_set_operation.get_level_set_as_heaviside(),
            level_set_operation.get_curvature(),
            base_in->parameters.flow.surface_tension_coefficient,
            ls_hanging_nodes_dof_idx,
            curv_dof_idx,
            flow_operation->get_dof_handler_idx_velocity(),
            flow_operation->get_quad_idx_velocity(),
            false /* false means not to zero out the force vector */);

        // ... c) temperature-dependent surface tension
        if (base_in->parameters.mp.do_heat_transfer &&
            std::abs(base_in->parameters.flow.temperature_dependent_surface_tension_coefficient) >
              0.0)
          {
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
              ls_hanging_nodes_dof_idx,
              curv_dof_idx,
              normal_dof_idx,
              vel_dof_idx,
              flow_operation->get_quad_idx_velocity(),
              temp_dof_idx,
              false /*false means add to force vector*/);
          }

        // .... d) evaporative mass fluxes
        if (evaporation_operation)
          {
            evaporation_operation->compute_mass_balance_source_term(
              mass_balance_rhs,
              flow_operation->get_dof_handler_idx_pressure(),
              flow_operation->get_quad_idx_pressure(),
              true /* zero out rhs */);
          }

        // ... e) recoil pressure forces
        if (melt_pool_operation)
          melt_pool_operation->compute_force_flow_rhs(
            vel_force_rhs, level_set_operation.get_level_set_as_heaviside(), false);

        //  ... and set the resulting forces within the Navier-Stokes solver
        flow_operation->set_force_rhs(vel_force_rhs);

        // Compute potential mass fluxes due to evaporation and set the corresponding rhs in
        // the mass balance equation
        if (evaporation_operation)
          flow_operation->set_mass_balance_rhs(mass_balance_rhs);

        // solver Navier-Stokes problem
        flow_operation->solve();

        Journal::print_formatted_norm(
          scratch_data->get_pcout(0),
          VectorTools::compute_L2_norm<dim>(flow_operation->get_velocity(),
                                            *scratch_data,
                                            flow_operation->get_dof_handler_idx_velocity(),
                                            flow_operation->get_quad_idx_velocity()),
          "velocity",
          "navier_stokes_adaflo",
          15 /*precision*/
        );
        Journal::print_formatted_norm(
          scratch_data->get_pcout(0),
          VectorTools::compute_L2_norm<dim>(flow_operation->get_pressure(),
                                            *scratch_data,
                                            flow_operation->get_dof_handler_idx_pressure(),
                                            flow_operation->get_quad_idx_pressure()),
          "pressure",
          "navier_stokes_adaflo",
          15 /*precision*/
        );
        // ... and output the results to vtk files.
        output_results(n, time_iterator.get_current_time(), base_in);

        if (base_in->parameters.amr.do_amr)
          refine_mesh(base_in);
      }
    Journal::print_end(scratch_data->get_pcout());
  }

  template <int dim>
  std::string
  MeltPoolProblem<dim>::get_name()
  {
    return "melt_pool";
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup DoFHandler
     */
    dof_handler.reinit(*base_in->triangulation);

    /*
     *  setup scratch data
     */
    scratch_data = std::make_shared<ScratchData<dim>>(base_in->mpi_communicator,
                                                      base_in->parameters.base.verbosity_level,
                                                      /*do_matrix_free*/ true);

    /*
     *  setup mapping
     */
    if (base_in->parameters.base.do_simplex)
      scratch_data->set_mapping(MappingFE<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree)));
    else
      scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));

    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler);

    ls_hanging_nodes_dof_idx = scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
    ls_dof_idx               = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
    reinit_dof_idx           = scratch_data->attach_constraint_matrix(reinit_constraints_dirichlet);
    temp_dof_idx             = scratch_data->attach_constraint_matrix(temp_constraints_dirichlet);

    /*
     *  create quadrature rule
     */
    if (base_in->parameters.base.do_simplex)
      {
        ls_quad_idx = scratch_data->attach_quadrature(
          QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
      }
    else
      {
        ls_quad_idx =
          scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));
      }

#ifdef MELT_POOL_DG_WITH_ADAFLO
    flow_operation = std::make_shared<AdafloWrapper<dim>>(*scratch_data, base_in);
    flow_vel_no_solid_dof_idx =
      scratch_data->attach_constraint_matrix(flow_velocity_constraints_no_solid);
    scratch_data->attach_dof_handler(flow_operation->get_dof_handler_velocity());
#else
    AssertThrow(false, ExcNotImplemented());
#endif
    /*
     *  set indices of flow dof handlers
     */
    vel_dof_idx      = flow_operation->get_dof_handler_idx_velocity();
    pressure_dof_idx = flow_operation->get_dof_handler_idx_pressure();

    setup_dof_system(base_in, false);

    /*
     *  initialize the time stepping scheme
     */
    time_iterator.initialize(
      TimeIteratorData<double>{base_in->parameters.time_stepping.start_time,
                               base_in->parameters.time_stepping.end_time,
                               base_in->parameters.time_stepping.time_step_size,
                               base_in->parameters.time_stepping.max_n_steps,
                               false /*cfl_condition-->not supported yet*/});
    /*
     *    initialize the levelset operation class
     *    and setup initial conditions
     */
    level_set_operation.initialize(scratch_data,
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
    if (base_in->parameters.mp.do_heat_transfer)
      heat_operation = std::make_shared<Heat::HeatTransferOperation<dim>>(
        base_in->get_bc("heat_transfer"),
        *scratch_data,
        base_in->parameters.heat,
        base_in->parameters.material,
        temp_dof_idx,
        temp_hanging_nodes_dof_idx,
        temp_quad_idx,
        vel_dof_idx,
        &flow_operation->get_velocity(),
        ls_hanging_nodes_dof_idx,
        &level_set_operation.get_level_set_as_heaviside());
    /*
     *    initialize the evaporation class
     */
    if (base_in->parameters.mp.do_evaporation)
      {
        evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim>>(
          scratch_data,
          level_set_operation.get_level_set_as_heaviside(),
          level_set_operation.get_normal_vector(),
          base_in,
          normal_dof_idx,
          vel_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx,
          &heat_operation->get_temperature(),
          temp_dof_idx);

        /*
         * register evaporative mass flux to compute the heat sink
         */
        heat_operation->register_evaporative_mass_flux(
          &evaporation_operation->get_evaporative_mass_flux(),
          base_in->parameters.evapor.latent_heat_of_evaporation);
      }
    /*
     *    initialize the melt pool operation class
     */
    if (base_in->parameters.mp.do_melt_pool)
      melt_pool_operation = std::make_shared<MeltPool::MeltPoolOperation<dim>>(
        scratch_data,
        base_in->parameters,
        ls_dof_idx,
        &heat_operation->get_temperature(),
        reinit_dof_idx,
        flow_operation->get_dof_handler_idx_velocity(),
        flow_vel_no_solid_dof_idx,
        flow_operation->get_quad_idx_velocity(),
        temp_hanging_nodes_dof_idx,
        base_in->parameters.time_stepping.start_time);

    if (base_in->parameters.heat.solidification)
      AssertThrow(
        melt_pool_operation,
        ExcMessage("If solidifcation is enabled the melt pool operation must be initialized! Check "
                   "if the parameter >>>do melt pool<<< is set to true. Abort..."));

    if (evaporation_operation)
      {
        evaporation_operation->register_evaporative_mass_flux_model(
          base_in->parameters.recoil,
          level_set_operation.get_distance_to_level_set(),
          base_in->parameters.reinit.constant_epsilon,
          base_in->parameters.reinit.scale_factor_epsilon);
      }
    /*
     *  set initial conditions of all operations
     */
    set_initial_condition(base_in);
    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(vel_dof_idx),
                                           base_in->parameters.paraview,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(vel_dof_idx),
                                           scratch_data->get_pcout(1));
    /*
     *    Do initial refinement steps if requested
     */
    if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
        {
          std::ostringstream str;
          str << "cycle: " << i << " n_dofs: " << dof_handler.n_dofs() << "(ls) + "
              << flow_operation->get_dof_handler_velocity().n_dofs() << "(vel) + "
              << flow_operation->get_dof_handler_pressure().n_dofs() << "(p)";

          if (heat_operation)
            str << " T.size " << heat_operation->get_temperature().size();
          if (melt_pool_operation)
            str << " solid.size " << melt_pool_operation->get_solid().size();

          Journal::print_line(scratch_data->get_pcout(), str.str(), "melt_pool_problem");
          refine_mesh(base_in);
          /*
           *  set initial conditions after initial AMR
           */
          set_initial_condition(base_in);
        }
    /*
     *  output results of initialization
     *  @todo: find a way to plot vectors on the refined mesh, which are only relevant for output
     *  and which must not be transferred to the new mesh everytime refine_mesh() is called.
     */
    output_results(0, base_in->parameters.time_stepping.start_time, base_in);
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::set_initial_condition(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /**
     *  set initial condition of the velocity field
     */
#ifdef MELT_POOL_DG_WITH_ADAFLO
    dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())
      ->set_initial_condition(*base_in->get_initial_condition("navier_stokes_u"));
#else
    AssertThrow(false, ExcNotImplemented());
#endif
    /*
     *  set initial conditions of the level set field
     */
    level_set_operation.set_initial_condition(*base_in->get_initial_condition("level_set"),
                                              flow_operation->get_velocity());
    /*
     *  set initial conditions of the temperature field
     *
     *  @todo: improve cases where it must not be specified
     */
    if ((heat_operation && !evaporation_operation && !melt_pool_operation) ||
        (evaporation_operation && !(base_in->parameters.evapor.evaporation_model == "constant")) ||
        (melt_pool_operation && !(base_in->parameters.laser.heat_source_model == "Analytical")))
      heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"));
    /*
     * set initial condition of the melt pool class
     */
    if (melt_pool_operation)
      {
        melt_pool_operation->set_initial_condition(level_set_operation.get_level_set_as_heaviside(),
                                                   level_set_operation.get_level_set());
      }
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                         const bool                           do_reinit)
  {
    if (base_in->parameters.base.do_simplex)
      {
        dof_handler.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
      }
    else
      {
        dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
      }
      /*
       *    initialize the flow operation class
       */
#ifdef MELT_POOL_DG_WITH_ADAFLO
    dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->reinit_1();
    flow_velocity_constraints_no_solid.copy_from(flow_operation->get_constraints_velocity());
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

    // periodic constraints
    for (const auto &bc : base_in->get_periodic_bc())
      {
        const auto [id_in, id_out, direction] = bc;
        DoFTools::make_periodicity_constraints(
          dof_handler, id_in, id_out, direction, ls_hanging_node_constraints);
      }

    // ize constraints
    ls_hanging_node_constraints.close();

    UtilityFunctions::check_constraints(dof_handler, ls_hanging_node_constraints);

    ls_constraints_dirichlet.close();
    ls_constraints_dirichlet.merge(
      ls_hanging_node_constraints,
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

    UtilityFunctions::check_constraints(dof_handler, ls_constraints_dirichlet);

    reinit_constraints_dirichlet.close();
    reinit_constraints_dirichlet.merge(
      ls_hanging_node_constraints,
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

    UtilityFunctions::check_constraints(dof_handler, reinit_constraints_dirichlet);

    temp_constraints_dirichlet.close();
    temp_constraints_dirichlet.merge(
      ls_hanging_node_constraints,
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

    UtilityFunctions::check_constraints(dof_handler, temp_constraints_dirichlet);

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
     *    initialize the rhs vector of the continuity equation
     */
    if (evaporation_operation)
      {
        evaporation_operation->reinit(); // @todo -- needed?
        scratch_data->initialize_dof_vector(mass_balance_rhs, pressure_dof_idx);
      }
    /*
     *    initialize the velocity for advecting the level set interface
     */
    scratch_data->initialize_dof_vector(interface_velocity, vel_dof_idx);
  }

  // todo: clean-up
  template <int dim>
  void
  MeltPoolProblem<dim>::update_phases(const VectorType &        ls_as_heaviside,
                                      const Parameters<double> &parameters) const
  {
    if (parameters.heat.solidification)
      melt_pool_operation->get_solid().update_ghost_values();

    double dummy;

    double mass = 0.0;

#if 0
    // compute the limit values of the material parameters
    const double max_density =
      parameters.heat.solidification ?
        std::max({parameters.material.solid.density,
                  parameters.material.first.density,
                  parameters.material.second.density}) :
        std::max(parameters.material.first.density, parameters.material.second.density);
    const double min_density =
      parameters.heat.solidification ?
        std::min({parameters.material.solid.density,
                  parameters.material.first.density,
                  parameters.material.second.density}) :
        std::min(parameters.material.first.density, parameters.material.second.density);
    const double max_viscosity =
      parameters.heat.solidification ?
        std::max({parameters.material.solid.viscosity,
                  parameters.material.first.viscosity,
                  parameters.material.second.viscosity}) :
        std::max(parameters.material.first.viscosity, parameters.material.second.viscosity);
    const double min_viscosity =
      parameters.heat.solidification ?
        std::min({parameters.material.solid.viscosity,
                  parameters.material.first.viscosity,
                  parameters.material.second.viscosity}) :
        std::min(parameters.material.first.viscosity, parameters.material.second.viscosity);
#endif

    scratch_data->get_matrix_free().template cell_loop<double, VectorType>(
      [&](const auto &matrix_free, auto &, const auto &ls_as_heaviside, auto macro_cells) {
        FECellIntegrator<dim, 1, double> ls_values(matrix_free,
                                                   ls_hanging_nodes_dof_idx,
                                                   flow_operation->get_quad_idx_velocity());
        FECellIntegrator<dim, 1, double> solid_values(matrix_free,
                                                      temp_hanging_nodes_dof_idx,
                                                      flow_operation->get_quad_idx_velocity());

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            ls_values.reinit(cell);
            ls_values.read_dof_values_plain(ls_as_heaviside);
            ls_values.evaluate(EvaluationFlags::values);

            if (parameters.heat.solidification)
              {
                solid_values.reinit(cell);
                solid_values.read_dof_values_plain(melt_pool_operation->get_solid());
                solid_values.evaluate(EvaluationFlags::values);
              }

            for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
              {
                const auto indicator = parameters.material.two_phase_properties_transition_type ==
                                           TwoPhasePropertiesTransitionType::sharp ?
                                         UtilityFunctions::heaviside(ls_values.get_value(q), 0.5) :
                                         ls_values.get_value(q);

                // set properties
                if (parameters.material.two_phase_properties_transition_type ==
                    TwoPhasePropertiesTransitionType::consistent_with_evaporation)
                  {
                    const double rho_g = parameters.material.first.density;
                    const double rho_l = parameters.material.second.density;
                    flow_operation->get_density(cell, q) =
                      rho_g / (1. + (rho_g / rho_l - 1) * indicator);
                  }
                else
                  flow_operation->get_density(cell, q) =
                    UtilityFunctions::interpolate(indicator,
                                                  parameters.material.first.density,
                                                  parameters.material.second.density);

                // set viscosity
                flow_operation->get_viscosity(cell, q) =
                  UtilityFunctions::interpolate(indicator,
                                                parameters.material.first.viscosity,
                                                parameters.material.second.viscosity);

                if (parameters.heat.solidification)
                  {
                    const auto &solid_fraction = solid_values.get_value(q);
                    if (!(solid_fraction == VectorizedArray<double>(0.0)))
                      {
                        if (solid_fraction == VectorizedArray<double>(1.0))
                          {
                            flow_operation->get_density(cell, q) =
                              parameters.material.solid.density;
                            flow_operation->get_viscosity(cell, q) =
                              parameters.material.solid.viscosity;
                          }
                        else
                          {
                            const auto fluid_density = flow_operation->get_density(cell, q);
                            flow_operation->get_density(cell, q) =
                              UtilityFunctions::interpolate_cubic(
                                solid_fraction, fluid_density, parameters.material.solid.density);
                            const auto fluid_viscosity = flow_operation->get_viscosity(cell, q);
                            flow_operation->get_viscosity(cell, q) =
                              UtilityFunctions::interpolate_cubic(
                                solid_fraction,
                                fluid_viscosity,
                                parameters.material.solid.viscosity);
                          }
                      }
                  }

#if 0
                // check if no spurious densities or viscosities are computed
                for (auto dens : flow_operation->get_density(cell, q))
                  if (min_density > dens || dens > max_density)
                    std::cout << "WARNING: density does not comply with input:" << dens
                              << std::endl;
                for (auto visc : flow_operation->get_viscosity(cell, q))
                  if (min_viscosity > visc || visc > max_viscosity)
                    std::cout << "WARNING: viscosity does not comply with input:" << visc
                              << std::endl;
#endif

                // compute overall mass
                for (unsigned int v = 0;
                     v < scratch_data->get_matrix_free().n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    mass += UtilityFunctions::interpolate(ls_values.get_value(q)[v],
                                                          parameters.material.first.density,
                                                          parameters.material.second.density) *
                            ls_values.JxW(q)[v];
                  }
              }
          }
      },
      dummy,
      ls_as_heaviside);

    if (evaporation_operation || melt_pool_operation)
      {
        std::ostringstream str;
        str << " total mass = " << std::setprecision(11)
            << Utilities::MPI::sum(mass, scratch_data->get_mpi_comm());
        Journal::print_line(scratch_data->get_pcout(), str.str(), "melt_pool_problem");
      }
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::compute_gravity_force(VectorType & vec,
                                              const double gravity,
                                              const bool   zero_out) const
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
            force_values.integrate_scatter(EvaluationFlags::values, vec);
          }
      },
      vec,
      nullptr,
      zero_out);
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::output_results(const unsigned int                   n_time_step,
                                       const double                         current_time,
                                       std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /**
     * collect all relevant output data
     */
    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      level_set_operation.attach_output_vectors(data_out);

      flow_operation->attach_output_vectors(data_out);

      if (melt_pool_operation)
        melt_pool_operation->attach_output_vectors(data_out);

      if (evaporation_operation)
        evaporation_operation->attach_output_vectors(data_out);
      if (heat_operation)
        heat_operation->attach_output_vectors(data_out);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(), current_time);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    base_in->do_postprocessing(generic_data_out);

    // paraview postprocessing
    post_processor->process(n_time_step, attach_output_vectors, current_time);
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement =
      [&](parallel::distributed::Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));

      locally_relevant_solution.copy_locally_owned_data_from(level_set_operation.get_level_set());
      ls_constraints_dirichlet.distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
        locally_relevant_solution.local_element(i) =
          (1.0 -
           locally_relevant_solution.local_element(i) * locally_relevant_solution.local_element(i));

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
        data.emplace_back(&flow_operation->get_dof_handler_velocity(),
                          [&](std::vector<VectorType *> &vectors) {
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
       * heat
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

  template class MeltPoolProblem<MELT_POOL_DG_DIM>;
} // namespace MeltPoolDG::Flow
