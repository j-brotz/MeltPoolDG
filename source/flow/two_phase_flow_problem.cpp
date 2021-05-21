#include <meltpooldg/flow/two_phase_flow_problem.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator.is_finished())
      {
        // In the melt pool simulations, the solid domain
        // can be considered as rigid by setting the constraints
        // for the velocity field to zero. If the solid domain
        // changes, the AffineConstraints for the velocity field
        // will be updated accordingly. In this case, also the
        // constrained indices in matrix-free have to be updated
        // which is done in the following by rebuilding matrix-free.
        //
        // @todo: alternative (better performing) solution?
        if (base_in->parameters.mp.set_velocity_to_zero_in_solid)
          scratch_data->build();

        const auto dt = time_iterator.get_next_time_increment();
        const auto n  = time_iterator.get_current_time_step_number();

        scratch_data->get_pcout() << "t= " << std::setw(10) << std::left
                                  << time_iterator.get_current_time();

        // ... solve level-set problem with the given advection field
        if (evaporation_operation)
          {
            /*
             If evaporative mass flux is considered the interface velocity will be modified.
             Note that the normal vector is used from the old step.
             */
            level_set_operation.update_normal_vector();

            //@todo: shift options to evaporation operation
            if (melt_pool_operation)
              evaporation_operation->compute_evaporative_mass_flux_from_temperature(
                melt_pool_operation->get_temperature(),
                temp_dof_idx,
                base_in->parameters.mp.boiling_temperature,
                base_in->parameters.recoil.pressure_constant,
                base_in->parameters.recoil.temperature_constant);
            else if (std::abs(base_in->parameters.evapor.evaporative_mass_flux) >
                     0.0) // constant value
              evaporation_operation->get_evaporative_mass_flux() =
                base_in->parameters.evapor.evaporative_mass_flux;
            else if (base_in->parameters.evapor.formulation_evaporative_mass_flux ==
                     "temperature dependent")
              evaporation_operation->compute_evaporative_mass_flux_from_temperature(
                heat_operation->get_temperature(), temp_dof_idx);
            else if (base_in->parameters.evapor.formulation_evaporative_mass_flux ==
                     "temperature dependent interface const")
              evaporation_operation
                ->compute_evaporative_mass_flux_from_temperature_const_over_interface(
                  level_set_operation.get_distance_to_level_set());

            if (base_in->parameters.evapor.formulation_source_term_continuity == "diffuse")
              evaporation_operation->compute_evaporation_velocity(
                base_in->parameters.flow.variable_properties_over_interface);
#ifdef MELT_POOL_DG_WITH_ADAFLO
            else if (base_in->parameters.evapor.formulation_source_term_continuity == "sharp")
              Evaporation::EvaporationOperationMarchingCube<dim>::compute_evaporation_velocity(
                *scratch_data,
                evaporation_operation->get_velocity(),
                evaporation_operation->get_evaporative_mass_flux(),
                level_set_operation.get_level_set_as_heaviside(),
                level_set_operation.get_normal_vector(),
                base_in->parameters.material.second.density,
                base_in->parameters.material.first.density,
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

        // solve heat problem
        if (!melt_pool_operation && heat_operation)
          {
            // solve only in case of temperature dependent evaporative mass flux
            if ((evaporation_operation &&
                 base_in->parameters.evapor.formulation_evaporative_mass_flux.find(
                   "temperature dependent") != std::string::npos) ||
                base_in->parameters.base.problem_name == "two_phase_flow_with_heat_transfer")
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
            ls_hanging_nodes_dof_idx,
            curv_dof_idx,
            flow_operation->get_dof_handler_idx_velocity(),
            flow_operation->get_quad_idx_velocity(),
            false /* false means not to zero out the vorce vector */);

        // ... c) temperature-dependent surface tension
        if ((melt_pool_operation || heat_operation) &&
            std::abs(base_in->parameters.flow.temperature_dependent_surface_tension_coefficient) >
              0.0)
          {
            // @todo: REFACTOR!!!
            const auto &temperature = (melt_pool_operation) ?
                                        melt_pool_operation->get_temperature() :
                                        heat_operation->get_temperature();
            Flow::SurfaceTensionOperation<dim>::compute_temperature_dependent_surface_tension(
              *scratch_data,
              vel_force_rhs,
              level_set_operation.get_level_set_as_heaviside(),
              level_set_operation.get_curvature(),
              temperature,
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

        if (evaporation_operation)
          {
            scratch_data->initialize_dof_vector(mass_balance_rhs, pressure_dof_idx);

            if (base_in->parameters.evapor.formulation_source_term_continuity == "diffuse")
              {
                evaporation_operation->compute_mass_balance_source_term(
                  mass_balance_rhs,
                  flow_operation->get_dof_handler_idx_pressure(),
                  flow_operation->get_quad_idx_pressure(),
                  true /* zero out force rhs */);
              }
#ifdef MELT_POOL_DG_WITH_ADAFLO
            else if (base_in->parameters.evapor.formulation_source_term_continuity == "sharp")
              Evaporation::EvaporationOperationMarchingCube<dim>::
                compute_mass_balance_source_term_sharp(
                  *scratch_data,
                  mass_balance_rhs,
                  evaporation_operation->get_evaporative_mass_flux(),
                  level_set_operation.get_level_set(),
                  base_in->parameters.material.second.density,
                  base_in->parameters.material.first.density,
                  ls_hanging_nodes_dof_idx,
                  flow_operation->get_dof_handler_idx_pressure());
#endif
            else
              AssertThrow(false, ExcNotImplemented());
          }

        // ... solve melt pool operation
        // It is assumed that material.first.density represents the density of the gas phase
        if (melt_pool_operation)
          {
            const double density_gas    = base_in->parameters.material.first.density;
            const double density_liquid = base_in->parameters.material.second.density;

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
        output_results(n, time_iterator.get_current_time(), base_in);

        if (base_in->parameters.amr.do_amr)
          refine_mesh(base_in);
      }
  }

  template <int dim>
  std::string
  TwoPhaseFlowProblem<dim>::get_name()
  {
    return "two_phase_flow";
  }

  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
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
    if (base_in->parameters.base.do_simplex)
      scratch_data->set_mapping(MappingFE<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree)));
    else
      scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));

    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler);
    scratch_data->attach_dof_handler(dof_handler_evapor);
    scratch_data->attach_dof_handler(dof_handler);

    ls_hanging_nodes_dof_idx = scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
    ls_dof_idx               = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
    reinit_dof_idx           = scratch_data->attach_constraint_matrix(reinit_constraints_dirichlet);
    evapor_vel_dof_idx = scratch_data->attach_constraint_matrix(evapor_hanging_node_constraints);
    temp_dof_idx       = scratch_data->attach_constraint_matrix(temp_constraints_dirichlet);

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
    time_iterator.initialize(TimeIteratorData<double>{base_in->parameters.flow.start_time,
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
    if (base_in->parameters.base.problem_name == "two_phase_flow_with_heat_transfer")
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
    else if (base_in->parameters.base.problem_name == "two_phase_flow_with_evaporation")
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
        &level_set_operation.get_level_set_as_heaviside(),
        &base_in->parameters.evapor);
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
          ls_quad_idx,
          &heat_operation->get_temperature(),
          temp_dof_idx);

        if (base_in->parameters.evapor.formulation_evaporative_mass_flux ==
            "temperature dependent interface const")
          heat_operation->register_evaporative_mass_flux(
            &evaporation_operation->get_evaporative_mass_flux());

        // configure also the level set problem with evaporation
        level_set_operation.setup_with_evaporation(flow_operation->get_dof_handler_idx_velocity(),
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
                                           scratch_data->get_triangulation(vel_dof_idx));
    /*
     *  output results of initialization --> initial refinement is done afterwards (!)
     *  @todo: find a way to plot vectors on the refined mesh, which are only relevant for output
     *  and which must not be transferred to the new mesh everytime refine_mesh() is called.
     */
    output_results(0, base_in->parameters.flow.start_time, base_in);
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
            scratch_data->get_pcout() << " T.size " << melt_pool_operation->get_temperature().size()
                                      << " solid.size " << melt_pool_operation->get_solid().size();
          if (heat_operation)
            scratch_data->get_pcout() << " T.size " << heat_operation->get_temperature().size();

          scratch_data->get_pcout() << std::endl;

          refine_mesh(base_in);
          /*
           *  set initial conditions after initial AMR
           */
          set_initial_condition(base_in);
        }
  }

  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::set_initial_condition(std::shared_ptr<SimulationBase<dim>> base_in)
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
     */
    if (base_in->parameters.base.problem_name == "two_phase_flow_with_heat_transfer" ||
        (base_in->parameters.base.problem_name == "two_phase_flow_with_evaporation" &&
         base_in->parameters.evapor.formulation_evaporative_mass_flux.find(
           "temperature dependent") != std::string::npos))
      heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"));
    /*
     * set initial condition of the melt pool class
     */
    if (melt_pool_operation)
      {
        const double density_gas    = base_in->parameters.material.first.density;
        const double density_liquid = base_in->parameters.material.second.density;

        melt_pool_operation->set_initial_condition(level_set_operation.get_level_set_as_heaviside(),
                                                   level_set_operation.get_level_set(),
                                                   density_gas,
                                                   density_liquid);
      }
  }

  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                             const bool                           do_reinit)
  {
    if (base_in->parameters.base.do_simplex)
      {
        dof_handler.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
        dof_handler_evapor.distribute_dofs(
          FESystem<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree), dim));
      }
    else
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
          dof_handler_evapor, id_in, id_out, direction, evapor_hanging_node_constraints);
      }

    // ize constraints
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
        evaporation_operation->reinit(); // @todo -- needed?
        scratch_data->initialize_dof_vector(mass_balance_rhs, pressure_dof_idx);
      }
  }

  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::update_phases(const VectorType &        src,
                                          const Parameters<double> &parameters) const
  {
    double dummy;

    double mass = 0.0;

    bool variable_props =
      parameters.flow.variable_properties_over_interface == "true" ||
      parameters.flow.variable_properties_over_interface == "consistent_with_evaporation";

    scratch_data->get_matrix_free().template cell_loop<double, VectorType>(
      [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
        FECellIntegrator<dim, 1, double> ls_values(matrix_free,
                                                   ls_hanging_nodes_dof_idx,
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
                    const double rho_g = parameters.material.first.density;
                    const double rho_l = parameters.material.second.density;
                    flow_operation->get_density(cell, q) =
                      rho_g / (1. + (rho_g / rho_l - 1) * ls_values.get_value(q));
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

                // check if no spurious densities or viscosities are computed
                const double min_density =
                  std::min(parameters.material.first.density, parameters.material.second.density);
                const double max_density =
                  std::max(parameters.material.first.density, parameters.material.second.density);

                for (auto dens : flow_operation->get_density(cell, q))
                  if (min_density > dens || dens > max_density)
                    std::cout << "WARNING: density does not comply with input:" << dens
                              << std::endl;

                const double min_viscosity = std::min(parameters.material.first.viscosity,
                                                      parameters.material.second.viscosity);
                const double max_viscosity = std::max(parameters.material.first.viscosity,
                                                      parameters.material.second.viscosity);

                for (auto visc : flow_operation->get_viscosity(cell, q))
                  if (min_viscosity > visc || visc > max_viscosity)
                    std::cout << "WARNING: viscosity does not comply with input:" << visc
                              << std::endl;

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
      src);

    if (evaporation_operation || melt_pool_operation)
      {
        scratch_data->get_pcout() << "    | two phase flow: total mass = "
                                  << Utilities::MPI::sum(mass, scratch_data->get_mpi_comm())
                                  << std::endl;
      }
  }

  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::compute_gravity_force(VectorType & vec,
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
            force_values.integrate_scatter(true, false, vec);
          }
      },
      vec,
      nullptr,
      zero_out);
  }

  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::output_results(const unsigned int                   n_time_step,
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

    GenericDataOut<dim> generic_data_out;
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    base_in->do_postprocessing(generic_data_out);

    // paraview postprocessing
    post_processor->process(n_time_step, attach_output_vectors, current_time);
  }

  template <int dim>
  void
  TwoPhaseFlowProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
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

  template class TwoPhaseFlowProblem<1>;
  template class TwoPhaseFlowProblem<2>;
  template class TwoPhaseFlowProblem<3>;
} // namespace MeltPoolDG::Flow
