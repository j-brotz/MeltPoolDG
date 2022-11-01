#ifndef MELT_POOL_DG_DIM
#  define MELT_POOL_DG_DIM 1
#endif

#include <deal.II/fe/fe_q_iso_q1.h>

#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/flow/surface_tension_operation.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/melt_pool/melt_pool_problem.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

namespace MeltPoolDG::MeltPool
{
  template <int dim>
  void
  MeltPoolProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in); // no timing needed, since the function does it self

    auto sc    = std::make_unique<ScopedName>("mp::run");
    auto scope = std::make_unique<TimerOutput::Scope>(scratch_data->get_timer(), *sc);

    while (!time_iterator->is_finished())
      {
        const auto dt = time_iterator->compute_next_time_increment();
        const int  n  = time_iterator->get_current_time_step_number();

        time_iterator->print_me(scratch_data->get_pcout());

        // use extrapolated solution values in the coupling terms
        if (problem_specific_parameters.do_extrapolate_coupling_terms)
          {
            if (flow_operation)
              flow_operation->init_time_advance();
            if (heat_operation)
              heat_operation->init_time_advance();
            if (level_set_operation)
              level_set_operation->init_time_advance();
          }

        // Only if a spatially constant evaporative mass flux is given as an analytical function,
        // the time is needed to evaluate the function.
        if (evaporation_operation &&
            base_in->parameters.evapor.evaporation_model == EvaporationModelType::constant)
          evaporation_operation->set_time(time_iterator->get_current_time());

        /******************************************************************************************
         * LEVEL SET
         ******************************************************************************************/
        if (problem_specific_parameters.do_advect_level_set)
          {
            ScopedName         sc("ls");
            TimerOutput::Scope scope(scratch_data->get_timer(), sc);

            TableHandler iter_table;
            VectorType   iter_res;

            const bool do_ls_iteration =
              problem_specific_parameters.level_set_evapor_coupling.n_max_iter > 1;

            if (do_ls_iteration)
              {
                scratch_data->initialize_dof_vector(iter_res, ls_dof_idx);
                iter_res = 1e10; // any large number
              }

            for (int i = 0; i < problem_specific_parameters.level_set_evapor_coupling.n_max_iter;
                 ++i)
              {
                if (do_ls_iteration)
                  {
                    iter_res -= level_set_operation->get_level_set();
                    const double res_norm = iter_res.l2_norm();

                    if (base_in->parameters.base.verbosity_level > 0)
                      {
                        iter_table.add_value("i", i);
                        iter_table.add_value("|res|", res_norm);
                        iter_table.add_value("|phi|",
                                             level_set_operation->get_level_set().l2_norm());
                      }

                    // early return of iteration if l2-norm of the change of the level set field is
                    // already very small
                    if (res_norm <= problem_specific_parameters.level_set_evapor_coupling.tol)
                      {
                        Journal::print_decoration_line(scratch_data->get_pcout(0));
                        Journal::print_line(scratch_data->get_pcout(0),
                                            "level set - evapor coupling; finished after " +
                                              std::to_string(i) + " iter at residual " +
                                              UtilityFunctions::to_string_with_precision(res_norm),
                                            "MeltPoolProblem");
                        Journal::print_decoration_line(scratch_data->get_pcout(0));
                        break;
                      }

                    iter_res.copy_locally_owned_data_from(level_set_operation->get_level_set());
                    Journal::print_decoration_line(scratch_data->get_pcout(1));
                    Journal::print_line(scratch_data->get_pcout(1),
                                        "level set - evapor coupling; #iter " + std::to_string(i),
                                        "MeltPoolProblem");
                    Journal::print_decoration_line(scratch_data->get_pcout(1));
                  }

                // TODO: remove; could not be removed since during
                // flow_operation->init_time_advance() an inconsistency in the DoF vectors is
                // introduced
                scratch_data->initialize_dof_vector(interface_velocity, vel_dof_idx);
                interface_velocity.copy_locally_owned_data_from(flow_operation->get_velocity());

                if (evaporation_operation &&
                    problem_specific_parameters.do_evaporative_velocity_jump)
                  {
                    ScopedName         sc("evaporation::level_set_source_term");
                    TimerOutput::Scope scope(scratch_data->get_timer(), sc);

                    switch (base_in->parameters.evapor.level_set_source_term_type)
                      {
                        default:
                          case EvaporationLevelSetSourceTermType::interface_velocity: {
                            // Option 1: compute modified advection velocity due to evaporation
                            if (problem_specific_parameters.do_extrapolate_coupling_terms)
                              {
                                level_set_operation->update_normal_vector();
                              }

                            evaporation_operation->compute_evaporation_velocity();
                            interface_velocity += evaporation_operation->get_velocity();
                            break;
                          }

                        case EvaporationLevelSetSourceTermType::rhs:
                          // Option 2: use source term as rhs in the level set equation
                          scratch_data->initialize_dof_vector(level_set_rhs, ls_dof_idx);
                          evaporation_operation->compute_level_set_source_term(
                            level_set_rhs,
                            ls_dof_idx,
                            level_set_operation->get_level_set(),
                            pressure_dof_idx);
                          level_set_operation->set_level_set_user_rhs(level_set_rhs);
                          break;
                      }
                  }

                // ... solve level-set problem with the given advection field
                scratch_data->get_constraint(vel_dof_idx).distribute(interface_velocity);

                level_set_operation->solve(false /*finish time step will be called later*/);

                if (do_ls_iteration)
                  {
                    level_set_operation->update_normal_vector();
                    level_set_operation->transform_level_set_to_smooth_heaviside();
                  }
              }

            if (base_in->parameters.base.verbosity_level > 0 && do_ls_iteration)
              {
                iter_table.set_precision("|res|", 10);
                iter_table.set_scientific("|res|", true);
                iter_table.set_precision("|phi|", 10);
                iter_table.set_scientific("|phi|", true);

                if (scratch_data->get_pcout(1).is_active())
                  iter_table.write_text(scratch_data->get_pcout(1).get_stream());

                Journal::print_decoration_line(scratch_data->get_pcout(1));
              }


            level_set_operation->finish_time_advance();

            if (evaporation_operation && base_in->parameters.evapor.formulation_source_term_heat ==
                                           InterfaceForceType::sharp)
              level_set_operation->update_surface_mesh();
          }

        /******************************************************************************************
         * HEAT TRANSFER
         ******************************************************************************************/
        {
          ScopedName         sc("heat");
          TimerOutput::Scope scope(scratch_data->get_timer(), sc);

          if (melt_pool_operation)
            melt_pool_operation->compute_heat_source(
              heat_operation->get_heat_source(),
              heat_operation->get_user_rhs(),
              level_set_operation->get_level_set_as_heaviside(),
              level_set_operation->get_normal_vector(),
              normal_dof_idx,
              dt,
              true /* zero_out */);

          // the heat equation will NOT be solved if
          //    * the evaporative mass flux is given as a constant (temperature-independent) value
          //    * the temperature field is prescribed analytically
          if ((heat_operation && !evaporation_operation && !melt_pool_operation) ||
              (evaporation_operation &&
               !(base_in->parameters.evapor.evaporation_model == EvaporationModelType::constant)) ||
              (melt_pool_operation &&
               !(base_in->parameters.laser.heat_source_model == LaserHeatSourceModel::Analytical)))
            {
              heat_operation->solve();
            }

          if (melt_pool_operation)
            {
              ScopedName         sc("melt_front_propagation");
              TimerOutput::Scope scope(scratch_data->get_timer(), sc);
              melt_pool_operation->compute_melt_front_propagation(
                level_set_operation->get_level_set_as_heaviside());

              if (base_in->parameters.mp.solid.set_velocity_to_zero ||
                  base_in->parameters.mp.solid.do_not_reinitialize)
                {
#ifdef MELT_POOL_DG_WITH_ADAFLO
                  dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())->reinit_3();
#else
                  AssertThrow(false, ExcNotImplemented());
#endif
                }
              // TODO zero_out only
              scratch_data->initialize_dof_vector(vel_force_rhs, vel_dof_idx);
            }
        }

        // compute the evaporative mass flux from the temperature field
        if (evaporation_operation)
          {
            ScopedName         sc("evaporation::mass_flux");
            TimerOutput::Scope scope(scratch_data->get_timer(), sc);
            evaporation_operation->compute_evaporative_mass_flux();
          }

        /******************************************************************************************
         * NAVIER - STOKES
         ******************************************************************************************/
        {
          ScopedName         sc("ns");
          TimerOutput::Scope scope(scratch_data->get_timer(), sc);
          {
            ScopedName         sc("compute_fluxes");
            TimerOutput::Scope scope(scratch_data->get_timer(), sc);
            // update the phases for the flow solver considering the updated level set and
            // temperature
            update_phases(level_set_operation->get_level_set_as_heaviside(), base_in->parameters);

            // ... a) gravity force
            compute_gravity_force(vel_force_rhs,
                                  base_in->parameters.base.gravity,
                                  true /* true means force vector is zeroed out before */);

            // ... b) (temperature-dependent) surface tension
            surface_tension_operation->compute_surface_tension(vel_force_rhs,
                                                               false /*do not zero out*/);

            if (base_in->parameters.surface_tension.time_step_limit.enable)
              {
                const auto dt_lim = surface_tension_operation->compute_time_step_limit(
                  base_in->parameters.material.first.density,
                  base_in->parameters.material.second.density);

                AssertThrow(time_iterator->check_time_step_limit(dt_lim),
                            ExcMessage("The time step limit for surface tension (dt=" +
                                       UtilityFunctions::to_string_with_precision(dt_lim) +
                                       ") is exceeded. Try to choose a smaller "
                                       "time step size. Abort..."));
              }


            // .... d) evaporative mass fluxes
            if (evaporation_operation && problem_specific_parameters.do_evaporative_velocity_jump)
              {
                evaporation_operation->compute_mass_balance_source_term(
                  mass_balance_rhs,
                  flow_operation->get_dof_handler_idx_pressure(),
                  flow_operation->get_quad_idx_pressure(),
                  true /* zero out rhs */);
              }

            // ... e) recoil pressure forces
            if (melt_pool_operation)
              {
                if (base_in->parameters.recoil.interface_distributed_flux_type ==
                    InterfaceDistributedFluxType::interface_value)
                  {
                    heat_operation->compute_interface_temperature(
                      level_set_operation->get_distance_to_level_set(),
                      level_set_operation->get_normal_vector());
                    melt_pool_operation->compute_force_flow_rhs(
                      vel_force_rhs,
                      level_set_operation->get_level_set_as_heaviside(),
                      heat_operation->get_temperature_interface(),
                      false);
                  }
                else
                  {
                    melt_pool_operation->compute_force_flow_rhs(
                      vel_force_rhs,
                      level_set_operation->get_level_set_as_heaviside(),
                      heat_operation->get_temperature(),
                      evaporation_operation->get_evaporative_mass_flux(),
                      false);
                  }
              }

            // ... f) explicit Darcy damping force
            if (darcy_operation && base_in->parameters.darcy.formulation ==
                                     DarcyDampingFormulation::explicit_formulation)
              darcy_operation->compute_darcy_damping(vel_force_rhs,
                                                     flow_operation->get_velocity(),
                                                     false /*zero_out*/);

            //  ... and set the resulting forces within the Navier-Stokes solver
            flow_operation->set_force_rhs(vel_force_rhs);
            // Compute potential mass fluxes due to evaporation and set the corresponding rhs in
            // the mass balance equation
            if (evaporation_operation && problem_specific_parameters.do_evaporative_velocity_jump)
              flow_operation->set_mass_balance_rhs(mass_balance_rhs);
          }

          {
            ScopedName         sc("solve");
            TimerOutput::Scope scope(scratch_data->get_timer(), sc);

            if (evaporation_fluid_material)
              evaporation_fluid_material->update_ghost_values();
            // solver Navier-Stokes problem
            flow_operation->solve();

            if (evaporation_fluid_material)
              evaporation_fluid_material->zero_out_ghost_values();
          }
        }

        {
          ScopedName         sc("output");
          TimerOutput::Scope scope(scratch_data->get_timer(), sc);

          // ... and output the results to vtk files.
          output_results(n, time_iterator->get_current_time(), base_in);
        }


        if (base_in->parameters.amr.do_amr)
          {
            ScopedName         sc("amr");
            TimerOutput::Scope scope(scratch_data->get_timer(), sc);
            refine_mesh(base_in);
          }

        //@todo: adapt in case of adaptive time stepping
        if ((n % base_in->parameters.profiling.write_frequency == 0) &&
            base_in->parameters.profiling.enable)
          {
            scratch_data->get_timer().print_wall_time_statistics(scratch_data->get_mpi_comm());
            scratch_data->get_pcout() << std::endl;
            IterationMonitor::print(scratch_data->get_pcout());
          }
      }
    Journal::print_end(scratch_data->get_pcout());

    scope.reset();
    sc.reset();

    //... always print timing statistics
    if (base_in->parameters.profiling.enable)
      {
        scratch_data->get_timer().print_wall_time_statistics(scratch_data->get_mpi_comm());
        scratch_data->get_pcout() << std::endl;
        IterationMonitor::print(scratch_data->get_pcout());
      }
  }

  template <int dim>
  std::string
  MeltPoolProblem<dim>::get_name()
  {
    return "melt_pool";
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("problem specific");
    {
      prm.add_parameter(
        "do heat transfer",
        problem_specific_parameters.do_heat_transfer,
        "Set this parameter to true if you want to consider a coupling with heat transfer.");
      prm.add_parameter(
        "do evaporative heat flux",
        problem_specific_parameters.do_evaporative_heat_flux,
        "Set this parameter to true if you want to the evaporation-induced jump in the enthalpy of the heat equation.");
      prm.add_parameter(
        "do evaporative velocity jump",
        problem_specific_parameters.do_evaporative_velocity_jump,
        "Set this parameter to true if you want to consider the evaporation-induced jump in the normal velocity. "
        "The latter is relevant for the source term in the continuity equation and the level set equation.");
      prm.add_parameter(
        "do recoil pressure",
        problem_specific_parameters.do_recoil_pressure,
        "Set this parameter to true to prescribe the evaporation-induced jump in the pressure field "
        "(i.e. recoil pressure), considered as an interfacial force in the momentum balance equation."
        "If 'do evaporative velocity jump' is set to true, this pressure jump will be added to the "
        "one resulting from the discontinuous normal velocity field.");
      prm.add_parameter(
        "do melt pool",
        problem_specific_parameters.do_melt_pool,
        "Set this parameter to true if you want to consider a melt pool simulation including solid/liquid/gaseous phases.");
      prm.add_parameter(
        "do advect level set",
        problem_specific_parameters.do_advect_level_set,
        "Set this parameter to true if you want to advect the level set with the fluid velocity.");
      prm.add_parameter(
        "do extrapolate coupling terms",
        problem_specific_parameters.do_extrapolate_coupling_terms,
        "Set this parameter to true if you want to extrapolate the solution vectors for semi-explicit "
        "treatment of coupling terms.");
      prm.enter_subsection("amr");
      {
        prm.add_parameter("strategy",
                          problem_specific_parameters.amr.strategy,
                          "Select the AMR strategy.");
        prm.add_parameter(
          "do auto detect frequency",
          problem_specific_parameters.amr.do_auto_detect_frequency,
          "Automatically determine the frequency of remeshing. If this parameter is set, the parameter "
          "`amr: every n step` is ignored.");
        prm.add_parameter(
          "automatic grid refinement type",
          problem_specific_parameters.amr.automatic_grid_refinement_type,
          "If the cells are refined automatically (strategy generic/KellyErrorEstimator), choose between "
          "refine_and_coarsen_fixed_number and refine_and_coarsen_fixed_fraction.");
        prm.add_parameter(
          "do refine all interface cells",
          problem_specific_parameters.amr.do_refine_all_interface_cells,
          "Enforce all cells with level set values between -0.975 and 0.975 to be refined.");
        prm.add_parameter(
          "fraction of melting point refined in solid",
          problem_specific_parameters.amr.fraction_of_melting_point_refined_in_solid,
          "Define a fraction of the melting point. Cells in the solid with a higher temperature are enforced "
          "to be refined.");
      }
      prm.leave_subsection();
      prm.enter_subsection("coupling ls evapor");
      {
        prm.add_parameter("n max iter",
                          problem_specific_parameters.level_set_evapor_coupling.n_max_iter,
                          "Maximum number of iterations for nonlinear solution.");
        prm.add_parameter("tol",
                          problem_specific_parameters.level_set_evapor_coupling.tol,
                          "If the change of the l2-norm of the level set is smaller than 'tol', "
                          "the iteration is stopped.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::check_input_parameters(
    Parameters<double> &parameters /*todo: could be made const*/)
  {
    AssertThrow(!problem_specific_parameters.do_recoil_pressure ||
                  (problem_specific_parameters.do_evaporative_velocity_jump ||
                   parameters.recoil.model_type == RecoilPressureModelType::phenomenological),
                ExcMessage("For the phenomenological recoil pressure model, no velocity jump "
                           "is allowed."));

    AssertThrow(!problem_specific_parameters.do_evaporative_heat_flux ||
                  parameters.material.latent_heat_of_evaporation > 0.0,
                ExcMessage("To consider the evaporative heat flux the value for "
                           ">>> latent heat of evaporation <<< "
                           "must be larger than zero."));

    if (problem_specific_parameters.do_melt_pool && !problem_specific_parameters.do_heat_transfer)
      AssertThrow(false,
                  ExcMessage("In case of do melt pool both flag >>> do melt pool <<< "
                             "and >>> do heat transfer <<< have to be set to true."));

    AssertThrow(problem_specific_parameters.amr.fraction_of_melting_point_refined_in_solid <= 1 &&
                  problem_specific_parameters.amr.fraction_of_melting_point_refined_in_solid >= 0,
                ExcMessage(
                  ">>>fraction of melting point refined in solid<<< must be between 0 and 1."));
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  parameters for adaflo
     */
#ifdef MELT_POOL_DG_WITH_ADAFLO
    base_in->parameters.adaflo_params.parse_parameters(base_in->parameter_file);
#endif
    /*
     *  Add problem specific parameters defined within add_parameters()
     */
    this->add_problem_specific_parameters(base_in->parameter_file);
    /*
     *  Check parameters for validity
     */
    check_input_parameters(base_in->parameters);
    /*
     *  setup DoFHandler
     */
    dof_handler_ls.reinit(*base_in->triangulation);
    dof_handler_heat.reinit(*base_in->triangulation);

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

    scratch_data->attach_dof_handler(dof_handler_ls);
    scratch_data->attach_dof_handler(dof_handler_ls);
    scratch_data->attach_dof_handler(dof_handler_ls);
    scratch_data->attach_dof_handler(dof_handler_ls);
    scratch_data->attach_dof_handler(dof_handler_heat);
    scratch_data->attach_dof_handler(dof_handler_heat);

    ls_hanging_nodes_dof_idx = scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
    ls_dof_idx               = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
    reinit_dof_idx           = scratch_data->attach_constraint_matrix(reinit_constraints_dirichlet);
    reinit_no_solid_dof_idx =
      scratch_data->attach_constraint_matrix(reinit_no_solid_constraints_dirichlet);
    temp_dof_idx = scratch_data->attach_constraint_matrix(temp_constraints_dirichlet);
    temp_hanging_nodes_dof_idx =
      scratch_data->attach_constraint_matrix(temp_hanging_node_constraints);

    /*
     *  create quadrature rule
     */
    if (base_in->parameters.base.do_simplex)
      {
        ls_quad_idx = scratch_data->attach_quadrature(
          QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
        temp_quad_idx = scratch_data->attach_quadrature(
          QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
      }
    else if (base_in->parameters.ls.n_subdivisions > 1)
      {
        ls_quad_idx = scratch_data->attach_quadrature(
          QIterated<dim>(QGauss<1>(2), base_in->parameters.ls.n_subdivisions));
        temp_quad_idx =
          scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));
      }
    else
      {
        ls_quad_idx =
          scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));
        temp_quad_idx =
          scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));
      }

    // initialize the time stepping scheme
    time_iterator = std::make_shared<TimeIterator<double>>(base_in->parameters.time_stepping);

    /*
     * initialize material
     */
    const auto material_type =
      determine_material_type(true,
                              base_in->parameters.heat.solidification,
                              base_in->parameters.material.two_phase_properties_transition_type ==
                                TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation);

    material = std::make_shared<Material<double>>(base_in->parameters.material, material_type);

#ifdef MELT_POOL_DG_WITH_ADAFLO
    flow_operation = std::make_shared<Flow::AdafloWrapper<dim>>(
      *scratch_data,
      base_in,
      *time_iterator,
      problem_specific_parameters.do_evaporative_velocity_jump);
    flow_vel_no_solid_dof_idx =
      scratch_data->attach_constraint_matrix(flow_velocity_constraints_no_solid);
    scratch_data->attach_dof_handler(flow_operation->get_dof_handler_velocity());
#else
    AssertThrow(false, ExcNotImplemented());
#endif
    /*
     *  set indices of flow dof handlers
     */
    vel_dof_idx = flow_operation->get_dof_handler_idx_velocity();

    pressure_dof_idx = flow_operation->get_dof_handler_idx_pressure();

    setup_dof_system(base_in, false);
    /*
     *    initialize the levelset operation class
     *    and setup initial conditions
     */
    level_set_operation =
      std::make_shared<LevelSet::LevelSetOperation<dim>>(*scratch_data,
                                                         *time_iterator,
                                                         base_in,
                                                         interface_velocity,
                                                         ls_dof_idx,
                                                         ls_hanging_nodes_dof_idx,
                                                         ls_quad_idx,
                                                         reinit_dof_idx,
                                                         curv_dof_idx,
                                                         normal_dof_idx,
                                                         vel_dof_idx,
                                                         ls_dof_idx /* todo: ls_zero_bc_idx*/);
    /*
     *    initialize the heat operation class
     */
    if (problem_specific_parameters.do_heat_transfer)
      heat_operation = std::make_shared<Heat::HeatTransferOperation<dim>>(
        base_in->get_bc("heat_transfer"),
        *scratch_data,
        base_in->parameters.heat,
        *material,
        *time_iterator,
        temp_dof_idx,
        temp_hanging_nodes_dof_idx,
        temp_quad_idx,
        vel_dof_idx,
        &flow_operation->get_velocity(),
        ls_hanging_nodes_dof_idx,
        &level_set_operation->get_level_set_as_heaviside());

    /*
     *    initialize the surface tension operation class
     */
    surface_tension_operation = std::make_shared<Flow::SurfaceTensionOperation<dim>>(
      base_in->parameters.surface_tension,
      *scratch_data,
      level_set_operation->get_level_set_as_heaviside(),
      level_set_operation->get_curvature(),
      ls_hanging_nodes_dof_idx,
      curv_dof_idx,
      vel_dof_idx,
      flow_operation->get_dof_handler_idx_pressure(),
      flow_operation->get_quad_idx_velocity());
    /*
     * Register temperature and normal vector in case of temperature dependent surface tension
     */
    if (heat_operation &&
        base_in->parameters.surface_tension.temperature_dependent_surface_tension_coefficient !=
          0.0)
      surface_tension_operation->register_temperature_and_normal_vector(
        temp_dof_idx,
        normal_dof_idx,
        &heat_operation->get_temperature(),
        &level_set_operation->get_normal_vector());

    /*
     *    initialize the evaporation class
     */
    if (problem_specific_parameters.do_evaporative_heat_flux ||
        problem_specific_parameters.do_evaporative_velocity_jump)
      {
        evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim>>(
          *scratch_data,
          level_set_operation->get_level_set_as_heaviside(),
          level_set_operation->get_normal_vector(),
          base_in,
          normal_dof_idx,
          evapor_vel_dof_idx,
          evapor_mass_flux_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx);
        /*
         * register temperature field
         */
        evaporation_operation->reinit(&heat_operation->get_temperature(),
                                      level_set_operation->get_distance_to_level_set(),
                                      base_in->parameters.recoil,
                                      base_in->parameters.reinit.constant_epsilon,
                                      base_in->parameters.reinit.scale_factor_epsilon,
                                      temp_dof_idx);

        /*
         *  Create a modified viscous stress-strain relation in case of an existing evaporation mass
         *  source term, such that div(u)!=0. This material law will be only evaluated if the
         * parameter "constitutive type" is set to "user defined" in the Navier-Stokes section.
         */

#ifdef MELT_POOL_DG_WITH_ADAFLO
        if (base_in->parameters.adaflo_params.params.constitutive_type ==
            FlowParameters::ConstitutiveType::user_defined)
          {
            evaporation_fluid_material = std::make_shared<
              Evaporation::IncompressibleNewtonianFluidEvaporationMaterial<dim, double>>(
              *scratch_data,
              [&](const unsigned int cell,
                  const unsigned int quad) -> const VectorizedArray<double> & {
                return flow_operation->get_viscosity(cell, quad);
              },
              level_set_operation->get_normal_vector(),
              level_set_operation->get_level_set_as_heaviside(),
              normal_dof_idx,
              ls_hanging_nodes_dof_idx,
              flow_operation->get_quad_idx_velocity());

            flow_operation->set_user_defined_material(
              [&](const Tensor<2, dim, VectorizedArray<double>> &velocity_grad,
                  const unsigned int                             cell,
                  const unsigned int                             q,
                  const bool do_tangent) -> Tensor<2, dim, VectorizedArray<double>> {
                evaporation_fluid_material->reinit(velocity_grad, cell, q);

                return do_tangent ? evaporation_fluid_material->get_vmult_d_tau_d_grad_vel() :
                                    evaporation_fluid_material->get_tau();
              });
          }
#endif

        /*
         * register evaporative mass flux to compute the heat sink
         */
        if (problem_specific_parameters.do_evaporative_heat_flux)
          {
            heat_operation->register_evaporative_mass_flux(
              &evaporation_operation->get_evaporative_mass_flux(),
              evapor_mass_flux_dof_idx,
              base_in->parameters.material.latent_heat_of_evaporation,
              problem_specific_parameters.do_recoil_pressure &&
                !problem_specific_parameters
                   .do_evaporative_velocity_jump /*do phenomenological recoil pressure model*/); // @todo:
                                                                                                 // clean-up
            if (base_in->parameters.evapor.formulation_source_term_heat ==
                InterfaceForceType::sharp)
              {
                heat_operation->register_surface_mesh(level_set_operation->get_surface_mesh_info());
              }
          }
      }
    /*
     *    initialize the melt pool operation class
     */
    if (problem_specific_parameters.do_melt_pool)
      {
        melt_pool_operation = std::make_shared<MeltPool::MeltPoolOperation<dim>>(
          *scratch_data,
          base_in->parameters,
          problem_specific_parameters.do_recoil_pressure,
          ls_hanging_nodes_dof_idx,
          &heat_operation->get_temperature(),
          reinit_dof_idx,
          reinit_no_solid_dof_idx,
          flow_operation->get_dof_handler_idx_velocity(),
          flow_vel_no_solid_dof_idx,
          flow_operation->get_quad_idx_velocity(),
          flow_operation->get_dof_handler_idx_pressure(),
          temp_dof_idx,
          temp_hanging_nodes_dof_idx,
          base_in->parameters.time_stepping.start_time);
        /*
         * Register solid fraction in surface tension
         */
        if (base_in->parameters.surface_tension.zero_surface_tension_in_solid)
          surface_tension_operation->register_solid_fraction(temp_hanging_nodes_dof_idx,
                                                             &melt_pool_operation->get_solid());
        /*
         *    initialize the darcy damping operation class
         */
        if (base_in->parameters.darcy.mushy_zone_morphology > 0.0)
          darcy_operation = std::make_shared<Flow::DarcyDampingOperation<dim>>(
            base_in->parameters.darcy,
            *scratch_data,
            flow_operation->get_dof_handler_idx_velocity(),
            flow_operation->get_quad_idx_velocity(),
            temp_hanging_nodes_dof_idx);
      }

    if (base_in->parameters.heat.solidification)
      AssertThrow(
        melt_pool_operation,
        ExcMessage("If solidifcation is enabled the melt pool operation must be initialized! Check "
                   "if the parameter >>>do melt pool<<< is set to true. Abort..."));

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
                                           base_in->parameters.time_stepping,
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
          str << "cycle: " << i << " n_dofs: " << dof_handler_ls.n_dofs() << "(ls) + "
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
    ScopedName         sc("mp::set_initial_condition");
    TimerOutput::Scope scope(scratch_data->get_timer(), sc);
    /**
     *  set initial condition of the velocity field
     */
#ifdef MELT_POOL_DG_WITH_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())
      ->set_initial_condition(*base_in->get_initial_condition("navier_stokes_u"));
#else
    AssertThrow(false, ExcNotImplemented());
#endif
    /*
     *  set initial conditions of the level set field ...
     */
    if (const auto initial_field =
          base_in->get_initial_condition("level_set", true /*is optional*/))
      {
        // ... via a given level set field
        level_set_operation->set_initial_condition(*initial_field);
      }
    else if (const auto initial_field =
               base_in->get_initial_condition("signed_distance", true /*is optional*/))
      {
        // ... or a given signed distance field.
        level_set_operation->set_initial_condition(*initial_field,
                                                   true /*is signed distance function*/);
      }
    else
      AssertThrow(
        false,
        ExcMessage("For the level set operation either a function for the initial level set or the "
                   "signed distance field must be provided. Abort ..."));
    /*
     *  set initial conditions of the temperature field
     *
     *  @todo: improve cases where it must not be specified
     */
    if ((heat_operation && !evaporation_operation && !melt_pool_operation) ||
        (evaporation_operation &&
         !(base_in->parameters.evapor.evaporation_model == EvaporationModelType::constant)) ||
        (melt_pool_operation &&
         !(base_in->parameters.laser.heat_source_model == LaserHeatSourceModel::Analytical)))
      heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"),
                                            base_in->parameters.time_stepping.start_time);

    /*
     * set initial condition of the melt pool class
     */
    if (melt_pool_operation)
      {
        melt_pool_operation->set_initial_condition(
          level_set_operation->get_level_set_as_heaviside(), level_set_operation->get_level_set());
        if (base_in->parameters.mp.solid.set_velocity_to_zero ||
            base_in->parameters.mp.solid.do_not_reinitialize)
#ifdef MELT_POOL_DG_WITH_ADAFLO
          dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())->reinit_3();
#else
          AssertThrow(false, ExcNotImplemented());
#endif
      }

    // update the phases for the flow solver considering the updated level set and temperature
    update_phases(level_set_operation->get_level_set_as_heaviside(), base_in->parameters);

    // compute the evaporative mass flux from the initial temperature field
    if (evaporation_operation)
      {
        // Only if a spatially constant evaporative mass flux is given as an analytical function,
        // the time is needed to evaluate the function.
        if (base_in->parameters.evapor.evaporation_model == EvaporationModelType::constant)
          evaporation_operation->set_time(time_iterator->get_current_time());

        evaporation_operation->compute_evaporative_mass_flux();
      }

    if (evaporation_operation &&
        base_in->parameters.evapor.formulation_source_term_heat == InterfaceForceType::sharp)
      level_set_operation->update_surface_mesh();
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                         const bool                           do_reinit)
  {
    if (base_in->parameters.base.do_simplex)
      {
        dof_handler_heat.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
        dof_handler_ls.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
      }
    else
      {
        if (base_in->parameters.ls.n_subdivisions > 1)
          dof_handler_ls.distribute_dofs(FE_Q_iso_Q1<dim>(base_in->parameters.ls.n_subdivisions));
        else
          dof_handler_ls.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));

        dof_handler_heat.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
      }

      /*
       *    initialize the flow operation class
       */
#ifdef MELT_POOL_DG_WITH_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())->reinit_1();
    flow_velocity_constraints_no_solid.copy_from(flow_operation->get_constraints_velocity());
#else
    AssertThrow(false, ExcNotImplemented());
#endif
    /*
     *  create partitioning
     */
    scratch_data->create_partitioning();
    /*
     * make constraints
     */
    //@todo move to a more central place
    base_in->attach_boundary_condition("level_set");
    base_in->attach_boundary_condition("reinitialization");
    base_in->attach_boundary_condition("heat_transfer");

    MeltPoolDG::UtilityFunctions::setup_constraints<dim>(*scratch_data,
                                                         base_in->get_dirichlet_bc("level_set"),
                                                         base_in->get_periodic_bc(),
                                                         ls_dof_idx,
                                                         ls_hanging_nodes_dof_idx);
    MeltPoolDG::UtilityFunctions::setup_and_merge_constraints<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("level_set"),
      reinit_dof_idx,
      ls_hanging_nodes_dof_idx,
      false /*set inhomogeneities to zero*/);

    // additional reinitialization dirichlet bc
    if (base_in->get_bc("reinitialization") &&
        !base_in->get_dirichlet_bc("reinitialization").get_data().empty())
      {
        for (const auto &bc : base_in->get_dirichlet_bc("reinitialization")
                                .get_data()) // @todo: add name of bc at a more central place
          {
            dealii::VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                             dof_handler_ls,
                                                             bc.first,
                                                             *bc.second,
                                                             reinit_constraints_dirichlet);
          }
      }
    reinit_constraints_dirichlet.close();
    reinit_no_solid_constraints_dirichlet.copy_from(reinit_constraints_dirichlet);

    MeltPoolDG::UtilityFunctions::setup_constraints<dim>(*scratch_data,
                                                         base_in->get_dirichlet_bc("heat_transfer"),
                                                         base_in->get_periodic_bc(),
                                                         temp_dof_idx,
                                                         temp_hanging_nodes_dof_idx);

    scratch_data->build();

    if (do_reinit)
      {
        level_set_operation->reinit();

        if (evaporation_operation)
          evaporation_operation->reinit();
        if (melt_pool_operation)
          melt_pool_operation->reinit();
        if (heat_operation)
          heat_operation->reinit();
      }

#ifdef MELT_POOL_DG_WITH_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())->reinit_2();
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
        scratch_data->initialize_dof_vector(level_set_rhs, ls_dof_idx);
      }
    /*
     *    initialize the velocity for advecting the level set interface
     */
    scratch_data->initialize_dof_vector(interface_velocity, vel_dof_idx);
    /*
     * print mesh information
     */
    Journal::print_mesh_information<dim>(*scratch_data, 1);
  }

  // todo: clean-up
  template <int dim>
  void
  MeltPoolProblem<dim>::update_phases(const VectorType &        ls_as_heaviside,
                                      const Parameters<double> &parameters) const
  {
    if (parameters.heat.solidification)
      heat_operation->get_temperature().update_ghost_values();

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
        FECellIntegrator<dim, 1, double> temp_values(matrix_free,
                                                     temp_dof_idx,
                                                     flow_operation->get_quad_idx_velocity());

        if (darcy_operation)
          darcy_operation->get_damping_at_q().resize(matrix_free.n_cell_batches(),
                                                     std::vector<VectorizedArray<double>>(
                                                       temp_values.n_q_points));

        const auto &material    = parameters.material;
        const auto  rho_g       = VectorizedArray<double>(material.first.density);
        const auto  viscosity_g = VectorizedArray<double>(material.first.viscosity);

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            ls_values.reinit(cell);
            ls_values.read_dof_values_plain(ls_as_heaviside);
            ls_values.evaluate(EvaluationFlags::values);

            if (parameters.heat.solidification)
              {
                temp_values.reinit(cell);
                temp_values.read_dof_values_plain(heat_operation->get_temperature());
                temp_values.evaluate(EvaluationFlags::values);
              }

            for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
              {
                const auto indicator = material.two_phase_properties_transition_type ==
                                           TwoPhaseFluidPropertiesTransitionType::sharp ?
                                         UtilityFunctions::heaviside(ls_values.get_value(q), 0.5) :
                                         ls_values.get_value(q);

                /*
                 * overwrite the liquid parameters in case of a solid/liquid mixture,
                 * e.g. rho_l = rho_ls(sf, rho_l, rho_s)
                 */
                auto rho_l       = VectorizedArray<double>(material.second.density);
                auto viscosity_l = VectorizedArray<double>(material.second.viscosity);

                if (parameters.heat.solidification)
                  {
                    const auto solid_fraction =
                      melt_pool_operation->compute_solid_fraction(temp_values.get_value(q));

                    rho_l = LevelSet::Tools::interpolate_cubic(solid_fraction,
                                                               material.second.density,
                                                               material.solid.density);


                    viscosity_l = LevelSet::Tools::interpolate_cubic(solid_fraction,
                                                                     material.second.viscosity,
                                                                     material.solid.viscosity);

                    if (darcy_operation)
                      {
                        darcy_operation->get_damping(cell, q) =
                          darcy_operation->get_darcy_damping_coefficient(solid_fraction *
                                                                         ls_values.get_value(q));
                        if (parameters.darcy.formulation ==
                            DarcyDampingFormulation::implicit_formulation)
                          flow_operation->get_damping(cell, q) =
                            darcy_operation->get_damping(cell, q);
                      }
                  }

                /*
                 * Interpolate the parameters for the two-phase flow system
                 * consisting of the two phases
                 *
                 *    gas <--> liquid
                 *
                 * or
                 *
                 *    gas <--> liquid/solid mixture
                 *
                 * in case of solidification.
                 */

                if (material.two_phase_properties_transition_type ==
                    TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation)
                  flow_operation->get_density(cell, q) =
                    LevelSet::Tools::interpolate_reciprocal(indicator, rho_g, rho_l);
                else
                  flow_operation->get_density(cell, q) =
                    LevelSet::Tools::interpolate(indicator, rho_g, rho_l);

                flow_operation->get_viscosity(cell, q) =
                  LevelSet::Tools::interpolate(indicator, viscosity_g, viscosity_l);
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
                const auto mass_contrib = flow_operation->get_density(cell, q) * ls_values.JxW(q);
                for (unsigned int v = 0;
                     v < scratch_data->get_matrix_free().n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    mass += mass_contrib[v];
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

#ifdef MELT_POOL_DG_WITH_ADAFLO
    if (parameters.adaflo_params.params.augmented_taylor_hood == true)
      {
        ls_as_heaviside.update_ghost_values();

        FEValues<dim> ls_values(
          scratch_data->get_mapping(),
          scratch_data->get_fe(ls_dof_idx),
          dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())->get_face_center_quad(),
          update_values);
        FEValues<dim> temp_values(
          scratch_data->get_mapping(),
          scratch_data->get_fe(temp_dof_idx),
          dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())->get_face_center_quad(),
          update_values);

        std::vector<double> hs(ls_values.n_quadrature_points);
        std::vector<double> temp(ls_values.n_quadrature_points);

        for (const auto &cell : scratch_data->get_triangulation().active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                TriaIterator<DoFCellAccessor<dim, dim, false>> ls_dof_cell(
                  &scratch_data->get_triangulation(),
                  cell->level(),
                  cell->index(),
                  &scratch_data->get_dof_handler(ls_dof_idx));

                ls_values.reinit(ls_dof_cell);
                ls_values.get_function_values(ls_as_heaviside, hs);

                const auto indicator = parameters.material.two_phase_properties_transition_type ==
                                           TwoPhaseFluidPropertiesTransitionType::sharp ?
                                         UtilityFunctions::heaviside(hs, 0.5) :
                                         hs;

                std::vector<double> rho_ls(ls_values.n_quadrature_points,
                                           parameters.material.second.density);

                if (parameters.heat.solidification)
                  {
                    TriaIterator<DoFCellAccessor<dim, dim, false>> temp_dof_cell(
                      &scratch_data->get_triangulation(),
                      cell->level(),
                      cell->index(),
                      &scratch_data->get_dof_handler(temp_dof_idx));

                    temp_values.reinit(temp_dof_cell);
                    temp_values.get_function_values(heat_operation->get_temperature(), temp);

                    for (unsigned int i = 0; i < rho_ls.size(); ++i)
                      rho_ls[i] = LevelSet::Tools::interpolate_cubic(
                        melt_pool_operation->compute_solid_fraction(temp[i]),
                        parameters.material.second.density,
                        parameters.material.solid.density);
                  }

                const double rho_g = parameters.material.first.density;

                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                  dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())
                    ->set_face_average_density(
                      cell,
                      f,
                      (parameters.material.two_phase_properties_transition_type ==
                       TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation) ?
                        LevelSet::Tools::interpolate_reciprocal(indicator[f], rho_g, rho_ls[f]) :
                        LevelSet::Tools::interpolate(indicator[f], rho_g, rho_ls[f]));
              }
          }
        ls_as_heaviside.zero_out_ghost_values();
      }
#endif

    if (parameters.heat.solidification)
      heat_operation->get_temperature().zero_out_ghost_values();
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
    if (!post_processor->now(n_time_step, current_time) &&
        !base_in->parameters.paraview.do_user_defined_postprocessing)
      return;
    /**
     * collect all relevant output data
     */
    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      level_set_operation->attach_output_vectors(data_out);

      flow_operation->attach_output_vectors(data_out);

      if (melt_pool_operation)
        melt_pool_operation->attach_output_vectors(data_out);

      if (evaporation_operation)
        evaporation_operation->attach_output_vectors(data_out);
      if (heat_operation)
        heat_operation->attach_output_vectors(data_out);
      if (darcy_operation)
        darcy_operation->attach_output_vectors(data_out);

      if (evaporation_operation && problem_specific_parameters.do_evaporative_velocity_jump &&
          base_in->parameters.evapor.level_set_source_term_type ==
            EvaporationLevelSetSourceTermType::interface_velocity)
        {
          /*
           *  interface velocity
           */
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            vector_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);

          data_out.add_data_vector(scratch_data->get_dof_handler(vel_dof_idx),
                                   interface_velocity,
                                   std::vector<std::string>(dim, "interface_velocity"),
                                   vector_component_interpretation);
        }
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
    const auto &amr_data = base_in->parameters.amr;

    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      level_set_operation->get_level_set().update_ghost_values();
      level_set_operation->get_normal_vector().update_ghost_values();

      if (problem_specific_parameters.amr.do_auto_detect_frequency)
        {
          // Check whether the interface changed that much such that refinement is needed.
          //
          // To this end, it is proved, if a cell K located within 3.5 cell layers around
          // the interface determined by
          //
          //          -log(max |∇Φ|ε) < 3.5
          //               K
          //
          // is at the maximum refinement level or not.
          std::vector<Point<1>> point(2);
          // For the level set gradient, look towards the end of the elements to find extrema in the
          // error indicator (= level set gradient).
          point[0][0] = 0.05;
          point[1][0] = 0.95;
          Quadrature<1>   quadrature_1d(point);
          Quadrature<dim> quadrature(quadrature_1d);

          FEValues<dim> ls_values(scratch_data->get_fe(ls_dof_idx), quadrature, update_values);
          FEValues<dim> vel_values(scratch_data->get_fe(vel_dof_idx), quadrature, update_values);

          // solution variables
          std::vector<std::vector<double>> ls_gradients(dim,
                                                        std::vector<double>(quadrature.size()));
          const double diffusion_length = (base_in->parameters.reinit.constant_epsilon > 0) ?
                                            base_in->parameters.reinit.constant_epsilon :
                                            scratch_data->get_min_cell_size() *
                                              base_in->parameters.reinit.scale_factor_epsilon /
                                              scratch_data->get_degree(ls_dof_idx);

          std::vector<double> ls_vals(quadrature.size());

          bool needs_refinement_or_coarsening = false;

          for (auto &cell : scratch_data->get_dof_handler(ls_dof_idx).active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  cell->clear_coarsen_flag();
                  cell->clear_refine_flag();

                  ls_values.reinit(cell);

                  for (unsigned int d = 0; d < dim; ++d)
                    ls_values.get_function_values(level_set_operation->get_normal_vector().block(d),
                                                  ls_gradients[d]);
                  double distance_in_cells = 0;
                  for (unsigned int q = 0; q < quadrature.size(); ++q)
                    {
                      Tensor<1, dim> ls_gradient;
                      for (unsigned int d = 0; d < dim; ++d)
                        ls_gradient[d] = ls_gradients[d][q];
                      distance_in_cells = std::max(distance_in_cells, ls_gradient.norm());
                    }

                  distance_in_cells = -std::log(distance_in_cells * diffusion_length);

                  if ((cell->level() < static_cast<int>(amr_data.max_grid_refinement_level) &&
                       distance_in_cells < 3.5) ||
                      (time_iterator->get_current_time_step_number() == 0 &&
                       cell->level() > amr_data.min_grid_refinement_level && distance_in_cells > 8))
                    {
                      needs_refinement_or_coarsening = true;
                      break;
                    }
                }
            }

          const unsigned int do_refine =
            Utilities::MPI::max(static_cast<unsigned int>(needs_refinement_or_coarsening),
                                scratch_data->get_mpi_comm());

          if (!do_refine)
            return false;
        }

      /*
       * different refinement strategies
       */

      switch (problem_specific_parameters.amr.strategy)
        {
            // Compute the error based on (1-level_set^2).
            case AMRStrategy::generic: {
              Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

              VectorType locally_relevant_solution;
              locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));

              locally_relevant_solution.copy_locally_owned_data_from(
                level_set_operation->get_level_set());
              ls_constraints_dirichlet.distribute(locally_relevant_solution);

              for (unsigned int i = 0; i < locally_relevant_solution.locally_owned_size(); ++i)
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

              switch (problem_specific_parameters.amr.automatic_grid_refinement_type)
                {
                  default: // this is the default case, since it was determined to be robust for CI
                           // testing
                    case AutomaticGridRefinementType::fixed_number: {
                      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
                        tria,
                        estimated_error_per_cell,
                        amr_data.upper_perc_to_refine,
                        amr_data.lower_perc_to_coarsen);
                      break;
                    }
                    case AutomaticGridRefinementType::fixed_fraction: {
                      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
                        tria,
                        estimated_error_per_cell,
                        amr_data.upper_perc_to_refine,
                        amr_data.lower_perc_to_coarsen);
                      break;
                    }
                }

              break;
            }
            case AMRStrategy::KellyErrorEstimator: {
              AssertThrow(
                base_in->parameters.ls.n_subdivisions <= 1,
                ExcMessage(
                  "For the KellyErrorEstimator n_subdivisions must not be larger than 1."));

              // 1) copy the solution
              VectorType locally_relevant_solution;
              locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));
              locally_relevant_solution.copy_locally_owned_data_from(
                level_set_operation->get_level_set());
              scratch_data->get_constraint(ls_dof_idx).distribute(locally_relevant_solution);
              locally_relevant_solution.update_ghost_values();

              Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

              // 2) estimate errors from the level set field
              KellyErrorEstimator<dim>::estimate(
                scratch_data->get_dof_handler(ls_dof_idx),
                scratch_data->get_face_quadrature(ls_dof_idx),
                std::map<types::boundary_id, const Function<dim> *>(),
                locally_relevant_solution,
                estimated_error_per_cell);

              // 3) optional: incorporate interface to solid in error estimator
              if (problem_specific_parameters.do_melt_pool)
                {
                  // 3a) copy the solution
                  locally_relevant_solution.reinit(
                    scratch_data->get_partitioner(temp_hanging_nodes_dof_idx));
                  locally_relevant_solution.copy_locally_owned_data_from(
                    melt_pool_operation->get_solid());
                  scratch_data->get_constraint(temp_hanging_nodes_dof_idx)
                    .distribute(locally_relevant_solution);
                  locally_relevant_solution.update_ghost_values();

                  // 3b) estimate errors from the solid
                  Vector<float> estimated_error_per_cell_solid(
                    base_in->triangulation->n_active_cells());
                  KellyErrorEstimator<dim>::estimate(
                    scratch_data->get_dof_handler(temp_hanging_nodes_dof_idx),
                    scratch_data->get_face_quadrature(temp_hanging_nodes_dof_idx),
                    {},
                    locally_relevant_solution,
                    estimated_error_per_cell_solid);
                  // 3c) merge two error indicators
                  for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
                    estimated_error_per_cell[i] =
                      std::max(estimated_error_per_cell[i], estimated_error_per_cell_solid[i]);
                }

              // 4) mark cells for refinement/coarsening
              switch (problem_specific_parameters.amr.automatic_grid_refinement_type)
                {
                  default: // this is the default case, since it was determined to be robust for CI
                           // testing
                    case AutomaticGridRefinementType::fixed_number: {
                      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
                        tria,
                        estimated_error_per_cell,
                        amr_data.upper_perc_to_refine,
                        amr_data.lower_perc_to_coarsen);
                      break;
                    }
                    case AutomaticGridRefinementType::fixed_fraction: {
                      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
                        tria,
                        estimated_error_per_cell,
                        amr_data.upper_perc_to_refine,
                        amr_data.lower_perc_to_coarsen);
                      break;
                    }
                }
              break;
            }
          case AMRStrategy::adaflo:
            // AMR strategy adopted from adaflo.
            //
            // Refine cell K if it is within four cell layers around the interface
            //
            //          -log(max |∇Φ|ε) < 4
            //                K
            //
            // or biased towards the flow direction
            //
            //                                 u*∇Φ
            //          -log(max |∇Φ|ε) - 4Δt ------  < 7
            //                K                |∇Φ|ε
            //
            // resulting in additional three cell layers, to reduce the re-meshing frequency.
            //
            // @todo: incorporate solid
            {
              std::vector<Point<1>> point(2);
              point[0][0] = 0.05;
              point[1][0] = 0.95;
              Quadrature<1>   quadrature_1d(point);
              Quadrature<dim> quadrature(quadrature_1d);

              FEValues<dim> ls_values(scratch_data->get_fe(ls_dof_idx), quadrature, update_values);
              FEValues<dim> vel_values(scratch_data->get_fe(vel_dof_idx),
                                       quadrature,
                                       update_values);
              std::vector<Tensor<1, dim>> vel_vals(quadrature.size());

              // solution variables
              std::vector<std::vector<double>> ls_gradients(dim,
                                                            std::vector<double>(quadrature.size()));
              const double diffusion_length = (base_in->parameters.reinit.constant_epsilon > 0) ?
                                                base_in->parameters.reinit.constant_epsilon :
                                                scratch_data->get_min_cell_size() *
                                                  base_in->parameters.reinit.scale_factor_epsilon /
                                                  scratch_data->get_degree(ls_dof_idx);

              std::vector<double> ls_vals(quadrature.size());

              const FEValuesExtractors::Vector velocity(0);

              typename DoFHandler<dim>::active_cell_iterator vel_cell =
                scratch_data->get_dof_handler(vel_dof_idx).begin_active();

              flow_operation->get_velocity().update_ghost_values();

              for (auto &cell : scratch_data->get_dof_handler(ls_dof_idx).active_cell_iterators())
                {
                  if (cell->is_locally_owned())
                    {
                      ls_values.reinit(cell);
                      vel_values.reinit(vel_cell);

                      ls_values.get_function_values(level_set_operation->get_level_set(), ls_vals);

                      for (unsigned int d = 0; d < dim; ++d)
                        ls_values.get_function_values(
                          level_set_operation->get_normal_vector().block(d), ls_gradients[d]);

                      double         distance_in_cells = 0;
                      Tensor<1, dim> ls_gradient;

                      for (unsigned int q = 0; q < quadrature.size(); ++q)
                        {
                          for (unsigned int d = 0; d < dim; ++d)
                            ls_gradient[d] = ls_gradients[d][q];
                          distance_in_cells = std::max(distance_in_cells, ls_gradient.norm());
                        }

                      distance_in_cells = -std::log(distance_in_cells * diffusion_length);


                      vel_values[velocity].get_function_values(flow_operation->get_velocity(),
                                                               vel_vals);

                      // try to look ahead and bias the error towards the flow direction
                      const double direction = 4. * time_iterator->get_current_time_increment() *
                                               (ls_gradient * vel_vals[0]) / ls_gradient.norm() /
                                               diffusion_length;
                      const double advected_distance_in_cells =
                        distance_in_cells - direction * ls_vals[0];

                      bool refine_cell =
                        ((cell->level() < static_cast<int>(amr_data.max_grid_refinement_level)) &&
                         (advected_distance_in_cells < 7 || distance_in_cells < 4));

                      if (refine_cell == true)
                        cell->set_refine_flag();
                      else if ((cell->level() > amr_data.min_grid_refinement_level) &&
                               (advected_distance_in_cells > 8 || distance_in_cells > 5))
                        cell->set_coarsen_flag();
                    }
                  vel_cell++;
                }

              flow_operation->get_velocity().zero_out_ghost_values();
              break;
            }
        }

      if (melt_pool_operation)
        {
          melt_pool_operation->get_liquid().update_ghost_values();
          melt_pool_operation->get_solid().update_ghost_values();
          heat_operation->get_temperature().update_ghost_values();
          Vector<double> liq_vals(
            scratch_data->get_fe(temp_hanging_nodes_dof_idx).n_dofs_per_cell());
          Vector<double> solid_vals(
            scratch_data->get_fe(temp_hanging_nodes_dof_idx).n_dofs_per_cell());
          Vector<double> temp_vals(
            scratch_data->get_fe(temp_hanging_nodes_dof_idx).n_dofs_per_cell());

          for (const auto &cell :
               scratch_data->get_dof_handler(temp_hanging_nodes_dof_idx).active_cell_iterators())
            {
              if (cell->is_locally_owned() == false)
                continue;

              cell->get_dof_values(melt_pool_operation->get_liquid(), liq_vals);
              cell->get_dof_values(melt_pool_operation->get_solid(), solid_vals);
              cell->get_dof_values(heat_operation->get_temperature(), temp_vals);

              for (unsigned int i = 0; i < liq_vals.size(); ++i)
                // ensure that the entire liquid region is refined
                if (liq_vals[i] > 0.0 && liq_vals[i] <= 1.0)
                  {
                    cell->clear_coarsen_flag();
                    cell->set_refine_flag();

                    break;
                  }
                // ensure that solid regions at high temperatures are refined
                else if (solid_vals[i] > 0.0 &&
                         temp_vals[i] >= problem_specific_parameters.amr
                                             .fraction_of_melting_point_refined_in_solid *
                                           base_in->parameters.material.melting_point)
                  {
                    cell->clear_coarsen_flag();
                    cell->set_refine_flag();
                    break;
                  }
            }
          melt_pool_operation->get_liquid().zero_out_ghost_values();
          melt_pool_operation->get_solid().zero_out_ghost_values();
          heat_operation->get_temperature().zero_out_ghost_values();
        }


      if (problem_specific_parameters.amr.do_refine_all_interface_cells)
        {
          // make sure that cells close to the interfaces are refined
          Vector<double> ls_vals(scratch_data->get_fe(ls_dof_idx).n_dofs_per_cell());
          for (const auto &cell : scratch_data->get_dof_handler(ls_dof_idx).active_cell_iterators())
            {
              if (cell->is_locally_owned() == false)
                continue;

              cell->get_dof_values(level_set_operation->get_level_set(), ls_vals);

              for (unsigned int i = 0; i < ls_vals.size(); ++i)
                // Ensure that values at -0.975<=phi<=0.975 are refined.
                // This includes cells in a maximum distance of ~4*epsilon to the interface.
                if (-0.975 <= ls_vals[i] && ls_vals[i] <= 0.975)
                  {
                    cell->clear_coarsen_flag();
                    cell->set_refine_flag();

                    break;
                  }
            }
        }

      level_set_operation->get_level_set().zero_out_ghost_values();
      level_set_operation->get_normal_vector().zero_out_ghost_values();
      return true;
    };

    // Early return if AMR is not needed according to auto detection
    if (problem_specific_parameters.amr.do_auto_detect_frequency)
      {
        auto triangulation = const_cast<Triangulation<dim> *>(&scratch_data->get_triangulation());
        if (!mark_cells_for_refinement(*triangulation))
          return;
      }

    /*
     * add DoFHandler and DoF-vector pairs to data
     */
    const auto attach_vectors =
      [&](std::vector<std::pair<const DoFHandler<dim> *,
                                std::function<void(std::vector<VectorType *> &)>>> &data) {
        data.emplace_back(&dof_handler_ls, [&](std::vector<VectorType *> &vectors) {
          level_set_operation->attach_vectors(vectors); // ls + heaviside
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
          data.emplace_back(&dof_handler_heat, [&](std::vector<VectorType *> &vectors) {
            melt_pool_operation->attach_vectors(vectors); // temperature + solid + liquid
          });

        if (evaporation_operation)
          {
            data.emplace_back(&flow_operation->get_dof_handler_velocity(),
                              [&](std::vector<VectorType *> &vectors) {
                                evaporation_operation->attach_dim_vectors(vectors);
                              });
            data.emplace_back(&dof_handler_heat, [&](std::vector<VectorType *> &vectors) {
              evaporation_operation->attach_vectors(vectors);
            });
          }

        if (heat_operation)
          data.emplace_back(&dof_handler_heat, [&](std::vector<VectorType *> &vectors) {
            heat_operation->attach_vectors(vectors);
          });
      };

    const auto post = [&]() {
      /**
       * level set
       */
      ls_constraints_dirichlet.distribute(level_set_operation->get_level_set());
      ls_hanging_node_constraints.distribute(level_set_operation->get_level_set_as_heaviside());

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
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 amr_data,
                                 time_iterator->get_current_time_step_number());
  }

  template class MeltPoolProblem<MELT_POOL_DG_DIM>;
} // namespace MeltPoolDG::MeltPool
