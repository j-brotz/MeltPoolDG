#ifndef MELT_POOL_DG_DIM
#  define MELT_POOL_DG_DIM 1
#endif

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>
#include <meltpooldg/evaporation/recoil_pressure_data.hpp>
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/heat/laser_analytical_temperature_field.hpp>
#include <meltpooldg/heat/laser_data.hpp>
#include <meltpooldg/interface/exceptions.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/material/material.templates.hpp>
#include <meltpooldg/material/material_data.hpp>
#include <meltpooldg/melt_pool/melt_pool_problem.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/cell_monitor.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/restart.templates.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <sstream>

namespace MeltPoolDG::MeltPool
{
  template <int dim>
  void
  MeltPoolProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in); // no timing needed, since the function does itself

    try
      {
        auto sc    = std::make_unique<ScopedName>("mp::run");
        auto scope = std::make_unique<TimerOutput::Scope>(scratch_data->get_timer(), *sc);

        if (restart_monitor && restart_monitor->do_load())
          {
            Journal::print_line(scratch_data->get_pcout(0), "load restart data");
            load(base_in);
          }
        /*
         *  output results of initialization
         *  @todo: find a way to plot vectors on the refined mesh, which are only relevant for
         * output and which must not be transferred to the new mesh everytime refine_mesh() is
         * called.
         */
        output_results(base_in);

        bool heat_up_finished = false;

        const auto heat_up_modify_time_step = [&]() {
          // check heat up
          if (heat_operation && problem_specific_parameters.mp_heat_up.time_step_size > 0 &&
              !heat_up_finished)
            {
              const double       T_max = heat_operation->get_temperature().linfty_norm();
              std::ostringstream s;
              s << "heat up period: T_max=" << std::setprecision(5) << std::scientific << T_max;
              Journal::print_line(scratch_data->get_pcout(0), s.str(), "melt_pool_problem");

              // start to decrease time step size if T > T_heat_up
              if (T_max > problem_specific_parameters.mp_heat_up.max_temperature)
                {
                  time_iterator->set_current_time_increment(
                    base_in->parameters.time_stepping.time_step_size,
                    problem_specific_parameters.mp_heat_up.max_change_factor_time_step_size);

                  // If time step size has decreased to the standard time stepping parameters, the
                  // heat up phase is finished.
                  heat_up_finished =
                    (std::abs(time_iterator->get_current_time_increment() -
                              base_in->parameters.time_stepping.time_step_size) < 1e-16);
                  if (heat_up_finished)
                    Journal::print_line(scratch_data->get_pcout(0),
                                        "Heat up period is finished.",
                                        "melt_pool_problem");
                }
            }
        };

        while (!time_iterator->is_finished())
          {
            heat_up_modify_time_step();

            const auto dt = time_iterator->compute_next_time_increment();

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

            // E.g. if a spatially constant evaporative mass flux is given as an analytical
            // function, the time is needed to evaluate the function.
            if (evaporation_operation)
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

                for (int i = 0;
                     i < problem_specific_parameters.level_set_evapor_coupling.n_max_iter;
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

                        // early return of iteration if l2-norm of the change of the level set field
                        // is already very small
                        if (res_norm <= problem_specific_parameters.level_set_evapor_coupling.tol)
                          {
                            Journal::print_decoration_line(scratch_data->get_pcout(0));
                            Journal::print_line(scratch_data->get_pcout(0),
                                                "level set - evapor coupling; finished after " +
                                                  std::to_string(i) + " iter at residual " +
                                                  UtilityFunctions::to_string_with_precision(
                                                    res_norm),
                                                "MeltPoolProblem");
                            Journal::print_decoration_line(scratch_data->get_pcout(0));
                            break;
                          }

                        iter_res.copy_locally_owned_data_from(level_set_operation->get_level_set());
                        Journal::print_decoration_line(scratch_data->get_pcout(1));
                        Journal::print_line(scratch_data->get_pcout(1),
                                            "level set - evapor coupling; #iter " +
                                              std::to_string(i),
                                            "MeltPoolProblem");
                        Journal::print_decoration_line(scratch_data->get_pcout(1));
                      }

                    this->compute_interface_velocity(base_in->parameters.ls,
                                                     base_in->parameters.evapor);

                    // ... solve level-set problem with the given advection field
                    scratch_data->get_constraint(vel_dof_idx).distribute(interface_velocity);

                    level_set_operation->solve(false /*finish time step will be called later*/);

                    if (do_ls_iteration)
                      {
                        level_set_operation->update_normal_vector();
                        level_set_operation->transform_level_set_to_smooth_heaviside();

                        // update evaporative mass flux if it is extrapolated from quantities at the
                        // interface
                        if (evaporation_operation &&
                            base_in->parameters.evapor.interface_temperature_evaluation_type !=
                              Evaporation::EvaporativeMassFluxTemperatureEvaluationType::
                                local_value)
                          evaporation_operation->compute_evaporative_mass_flux();
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

                if (evaporation_operation &&
                    (base_in->parameters.evapor.evaporative_cooling.model ==
                       Evaporation::EvaporCoolingInterfaceFluxType::sharp ||
                     base_in->parameters.evapor.evaporative_dilation_rate.model ==
                       Evaporation::InterfaceFluxType::sharp))
                  level_set_operation->update_surface_mesh();
              }

            /******************************************************************************************
             * HEAT TRANSFER
             ******************************************************************************************/
            if (heat_operation)
              {
                ScopedName         sc("heat");
                TimerOutput::Scope scope(scratch_data->get_timer(), sc);

                if (laser_operation)
                  {
                    laser_operation->move_laser(dt);

                    if (base_in->parameters.laser.model ==
                        Heat::LaserModelType::analytical_temperature)
                      Heat::LaserAnalyticalTemperatureField<dim>::compute_temperature_field(
                        *scratch_data,
                        base_in->parameters.material,
                        base_in->parameters.laser,
                        laser_operation->get_laser_power(),
                        laser_operation->get_laser_position(),
                        heat_operation->get_temperature(),
                        level_set_operation->get_level_set_as_heaviside(),
                        temp_dof_idx);
                    else
                      laser_operation->compute_heat_source(
                        heat_operation->get_heat_source(),
                        heat_operation->get_user_rhs(),
                        level_set_operation->get_level_set_as_heaviside(),
                        ls_dof_idx,
                        temp_hanging_nodes_dof_idx,
                        temp_quad_idx,
                        true /* zero_out */,
                        &level_set_operation->get_normal_vector(),
                        normal_dof_idx);
                  }
                TableHandler iter_table;
                VectorType   iter_res;

                const bool do_heat_iteration =
                  problem_specific_parameters.heat_evapor_coupling.n_max_iter > 1 &&
                  evaporation_operation;

                if (do_heat_iteration)
                  {
                    scratch_data->initialize_dof_vector(iter_res, temp_dof_idx);
                    iter_res = 1e10; // any large number
                  }

                for (int i = 0; i < problem_specific_parameters.heat_evapor_coupling.n_max_iter;
                     ++i)
                  {
                    if (do_heat_iteration)
                      {
                        iter_res -= heat_operation->get_temperature();
                        const double res_norm = iter_res.l2_norm();

                        if (base_in->parameters.base.verbosity_level > 0)
                          {
                            iter_table.add_value("i", i);
                            iter_table.add_value("|res|", res_norm);
                            iter_table.add_value("|T|",
                                                 heat_operation->get_temperature().l2_norm());
                          }

                        // early return of iteration if l2-norm of the change of the level set
                        // field is already very small
                        if (res_norm <= problem_specific_parameters.heat_evapor_coupling.tol)
                          {
                            Journal::print_decoration_line(scratch_data->get_pcout(0));
                            Journal::print_line(scratch_data->get_pcout(0),
                                                "heat - evapor coupling; finished after " +
                                                  std::to_string(i) + " iter at residual " +
                                                  UtilityFunctions::to_string_with_precision(
                                                    res_norm),
                                                "MeltPoolProblem");
                            Journal::print_decoration_line(scratch_data->get_pcout(0));
                            break;
                          }

                        iter_res.copy_locally_owned_data_from(heat_operation->get_temperature());
                        Journal::print_decoration_line(scratch_data->get_pcout(1));
                        Journal::print_line(scratch_data->get_pcout(1),
                                            "heat - evapor coupling; #iter " + std::to_string(i),
                                            "MeltPoolProblem");
                        Journal::print_decoration_line(scratch_data->get_pcout(1));
                      }


                    // the heat equation will NOT be solved if
                    //    * the evaporative mass flux is given as a constant
                    //    (temperature-independent) value
                    //    * the temperature field is prescribed analytically
                    if (heat_operation)
                      {
                        if (!((evaporation_operation &&
                               (base_in->parameters.evapor.evaporative_mass_flux_model ==
                                Evaporation::EvaporationModelType::analytical)) ||
                              base_in->parameters.laser.model ==
                                Heat::LaserModelType::analytical_temperature))
                          {
                            heat_operation->solve(false);
                          }
                      }

                    if (do_heat_iteration)
                      {
                        evaporation_operation->compute_evaporative_mass_flux();
                      }
                  }

                if (base_in->parameters.base.verbosity_level > 0 && do_heat_iteration)
                  {
                    iter_table.set_precision("|res|", 10);
                    iter_table.set_scientific("|res|", true);
                    iter_table.set_precision("|T|", 10);
                    iter_table.set_scientific("|T|", true);

                    if (scratch_data->get_pcout(1).is_active() &&
                        Utilities::MPI::this_mpi_process(scratch_data->get_mpi_comm()) == 0)
                      iter_table.write_text(std::cout);

                    Journal::print_decoration_line(scratch_data->get_pcout(1));
                  }

                heat_operation->finish_time_advance();



                if (melt_front_propagation)
                  {
                    ScopedName         sc("melt_front_propagation");
                    TimerOutput::Scope scope(scratch_data->get_timer(), sc);
                    melt_front_propagation->compute_melt_front_propagation(
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
                set_phase_dependent_parameters_flow(base_in->parameters);

                // ... a) gravity force
                compute_gravity_force(vel_force_rhs,
                                      base_in->parameters.flow.gravity,
                                      true /* true means force vector is zeroed out before */);

                // ... b) (temperature-dependent) surface tension
                surface_tension_operation->compute_surface_tension(vel_force_rhs,
                                                                   false /*do not zero out*/);

                if (base_in->parameters.flow.surface_tension.time_step_limit.enable)
                  {
                    const auto dt_lim = surface_tension_operation->compute_time_step_limit(
                      base_in->parameters.material.gas.density,
                      base_in->parameters.material.liquid.density);

                    AssertThrow(time_iterator->check_time_step_limit(dt_lim),
                                ExcMessage("The time step limit for surface tension (dt=" +
                                           UtilityFunctions::to_string_with_precision(dt_lim) +
                                           ") is exceeded. Try to choose a smaller "
                                           "time step size. Abort..."));
                  }


                // .... d) evaporative mass fluxes
                if (evaporation_operation &&
                    base_in->parameters.evapor.evaporative_dilation_rate.enable)
                  {
                    evaporation_operation->compute_mass_balance_source_term(
                      mass_balance_rhs,
                      flow_operation->get_dof_handler_idx_pressure(),
                      flow_operation->get_quad_idx_pressure(),
                      true /* zero out rhs */);
                  }

                // ... e) recoil pressure forces
                if (recoil_pressure_operation)
                  {
                    if (base_in->parameters.evapor.recoil.interface_distributed_flux_type ==
                        Evaporation::RegularizedRecoilPressureTemperatureEvaluationType::
                          interface_value)
                      {
                        heat_operation->compute_interface_temperature(
                          level_set_operation->get_distance_to_level_set(),
                          level_set_operation->get_normal_vector(),
                          base_in->parameters.ls.nearest_point);

                        recoil_pressure_operation->compute_recoil_pressure_force(
                          vel_force_rhs,
                          level_set_operation->get_level_set_as_heaviside(),
                          heat_operation->get_temperature_interface(),
                          evaporation_operation->get_evaporative_mass_flux(),
                          evapor_mass_flux_dof_idx,
                          false /*false means add to force vector*/);
                      }
                    else
                      {
                        recoil_pressure_operation->compute_recoil_pressure_force(
                          vel_force_rhs,
                          level_set_operation->get_level_set_as_heaviside(),
                          heat_operation->get_temperature(),
                          evaporation_operation->get_evaporative_mass_flux(),
                          evapor_mass_flux_dof_idx,
                          false /*false means add to force vector*/);
                      }
                  }

                // ... f) explicit Darcy damping force
                if (darcy_operation && base_in->parameters.flow.darcy_damping.formulation ==
                                         DarcyDampingFormulation::explicit_formulation)
                  darcy_operation->compute_darcy_damping(vel_force_rhs,
                                                         flow_operation->get_velocity(),
                                                         false /*zero_out*/);

                //  ... and set the resulting forces within the Navier-Stokes solver
                flow_operation->set_force_rhs(vel_force_rhs);
                // Compute potential mass fluxes due to evaporation and set the corresponding rhs in
                // the mass balance equation
                if (evaporation_operation &&
                    base_in->parameters.evapor.evaporative_dilation_rate.enable)
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
              output_results(base_in);
            }

            if (base_in->parameters.amr.do_amr)
              {
                ScopedName         sc("amr");
                TimerOutput::Scope scope(scratch_data->get_timer(), sc);
                refine_mesh(base_in);
              }

            if (profiling_monitor && profiling_monitor->now())
              {
                // call destructor as a workaround to also print the accumalated
                // time of mp::run at this stage
                scope.reset();
                sc.reset();
                sc    = std::make_unique<ScopedName>("mp::run");
                scope = std::make_unique<TimerOutput::Scope>(scratch_data->get_timer(), *sc);

                profiling_monitor->print(scratch_data->get_pcout(),
                                         scratch_data->get_timer(),
                                         scratch_data->get_mpi_comm());
              }

            if (restart_monitor && restart_monitor->do_save())
              {
                Journal::print_line(scratch_data->get_pcout(), "save restart data");
                restart_monitor->prepare_save();
                save(base_in);
              }
          }

        Journal::print_end(scratch_data->get_pcout());

        scope.reset();
        sc.reset();

        //... always print timing statistics
        if (profiling_monitor)
          {
            profiling_monitor->print(scratch_data->get_pcout(),
                                     scratch_data->get_timer(),
                                     scratch_data->get_mpi_comm());
          }
      }
    catch (const ExcHeatTransferNoConvergence &e)
      {
        finalize(base_in, OutputNotConvergedOperation::heat_transfer);
        AssertThrow(false, e);
      }
#ifdef MELT_POOL_DG_WITH_ADAFLO
    catch (const ExcNavierStokesNoConvergence &e)
      {
        finalize(base_in, OutputNotConvergedOperation::navier_stokes);
        AssertThrow(false, e);
      }
#endif
    catch (const ExcNewtonDidNotConverge &e)
      {
        finalize(base_in);
        AssertThrow(false, e);
      }
    catch (const SolverControl::NoConvergence &e)
      {
        finalize(base_in);
        AssertThrow(false, e);
      }
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::save(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    std::ofstream ofs(base_in->parameters.restart.prefix + "_0_problem.restart");
    {
      boost::archive::text_oarchive oa(ofs);
      oa << *time_iterator;
      oa << *post_processor;
    }

    const auto attach_vectors =
      [&](std::vector<std::pair<const DoFHandler<dim> *,
                                std::function<void(std::vector<VectorType *> &)>>> &data) {
        this->attach_vectors(data);
      };

    Restart::serialize_internal<dim, VectorType>(attach_vectors,
                                                 base_in->parameters.restart.prefix + "_0");
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::load(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const std::string load_prefix =
      base_in->parameters.restart.prefix + "_" + std::to_string(base_in->parameters.restart.load);

    AssertThrow(std::filesystem::exists(load_prefix + "_problem.restart"),
                ExcMessage("You tried to load the following "
                           "restart prefix '" +
                           load_prefix +
                           "'. However, it could not be found. Did you specify a wrong value "
                           "for the parameter 'load'?"));

    std::ifstream ifs(load_prefix + "_problem.restart");
    {
      boost::archive::text_iarchive ia(ifs);
      ia >> *time_iterator;
      ia >> *post_processor;
    }

    const auto attach_vectors =
      [&](std::vector<std::pair<const DoFHandler<dim> *,
                                std::function<void(std::vector<VectorType *> &)>>> &data) {
        this->attach_vectors(data);
      };

    const auto post = [&]() { this->post(); };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

    Restart::deserialize_internal<dim, VectorType>(attach_vectors,
                                                   post,
                                                   setup_dof_system,
                                                   load_prefix);
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
        "do solidification",
        problem_specific_parameters.do_solidification,
        "Set this parameter to true if you want to consider melting/solidification effects.");
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
        prm.add_parameter("refine gas domain",
                          problem_specific_parameters.amr.refine_gas_domain,
                          "Refine the gas domain.");
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
      prm.enter_subsection("mp heat up");
      {
        prm.add_parameter("time step size",
                          problem_specific_parameters.mp_heat_up.time_step_size,
                          "Time step size until heat up is finished.");
        prm.add_parameter(
          "max change factor time step size",
          problem_specific_parameters.mp_heat_up.max_change_factor_time_step_size,
          "Maximum allowed factor of changing the time step size between two time steps.");
        prm.add_parameter("max temperature",
                          problem_specific_parameters.mp_heat_up.max_temperature,
                          "Temperature at which heat up is finished.");
      }
      prm.leave_subsection();
      prm.enter_subsection("coupling heat evapor");
      {
        prm.add_parameter("n max iter",
                          problem_specific_parameters.heat_evapor_coupling.n_max_iter,
                          "Maximum number of iterations for nonlinear solution.");
        prm.add_parameter("tol",
                          problem_specific_parameters.heat_evapor_coupling.tol,
                          "If the change of the l2-norm of the level set is smaller than 'tol', "
                          "the iteration is stopped.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::check_input_parameters()
  {
    if (problem_specific_parameters.do_solidification &&
        !problem_specific_parameters.do_heat_transfer)
      AssertThrow(false,
                  ExcMessage("In case of solidification flag >>> do solidification <<< "
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
    check_input_parameters();
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
    scratch_data->set_mapping(FiniteElementUtils::create_mapping<dim>(base_in->parameters.base.fe));

    scratch_data->attach_dof_handler(dof_handler_ls);
    scratch_data->attach_dof_handler(dof_handler_ls);
    scratch_data->attach_dof_handler(dof_handler_ls);
    scratch_data->attach_dof_handler(dof_handler_ls);

    ls_hanging_nodes_dof_idx = scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
    ls_dof_idx               = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
    reinit_dof_idx           = scratch_data->attach_constraint_matrix(reinit_constraints_dirichlet);
    reinit_no_solid_dof_idx =
      scratch_data->attach_constraint_matrix(reinit_no_solid_constraints_dirichlet);
    if (problem_specific_parameters.do_heat_transfer)
      {
        scratch_data->attach_dof_handler(dof_handler_heat);
        scratch_data->attach_dof_handler(dof_handler_heat);

        temp_dof_idx = scratch_data->attach_constraint_matrix(temp_constraints_dirichlet);
        temp_hanging_nodes_dof_idx =
          scratch_data->attach_constraint_matrix(temp_hanging_node_constraints);
      }

    /*
     *  create quadrature rule
     */
    ls_quad_idx = scratch_data->attach_quadrature(
      FiniteElementUtils::create_quadrature<dim>(base_in->parameters.ls.fe));
    if (problem_specific_parameters.do_heat_transfer)
      temp_quad_idx = scratch_data->attach_quadrature(
        FiniteElementUtils::create_quadrature<dim>(base_in->parameters.heat.fe));

    // initialize the time stepping scheme
    time_iterator = std::make_shared<TimeIterator<double>>(base_in->parameters.time_stepping);

    if (problem_specific_parameters.mp_heat_up.time_step_size > 0)
      time_iterator->set_current_time_increment(
        problem_specific_parameters.mp_heat_up.time_step_size);


    /*
     * initialize material
     */
    const auto material_type = determine_material_type(
      true,
      problem_specific_parameters.do_solidification,
      base_in->parameters.material.two_phase_fluid_properties_transition_type ==
        TwoPhaseFluidPropertiesTransitionType::consistent_with_evaporation);

    material = std::make_shared<Material<double>>(base_in->parameters.material, material_type);

#ifdef MELT_POOL_DG_WITH_ADAFLO
    flow_operation = std::make_shared<Flow::AdafloWrapper<dim>>(
      *scratch_data,
      base_in,
      *time_iterator,
      base_in->parameters.evapor.evaporative_cooling.enable);
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
     * laser operation
     */
    if (base_in->parameters.laser.power > 0.0)
      {
        laser_operation = std::make_shared<Heat::LaserOperation<dim>>(
          *scratch_data,
          base_in->get_periodic_bc(),
          base_in->parameters,
          &level_set_operation->get_level_set_as_heaviside(),
          ls_dof_idx);
        laser_operation->reset(base_in->parameters.time_stepping.start_time);
      }

    setup_dof_system(base_in, false);

    level_set_operation->reinit();

    if (laser_operation)
      laser_operation->reinit();

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
        &level_set_operation->get_level_set_as_heaviside(),
        problem_specific_parameters.do_solidification);

    /*
     *    initialize the surface tension operation class
     */
    surface_tension_operation = std::make_shared<Flow::SurfaceTensionOperation<dim>>(
      base_in->parameters.flow.surface_tension,
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
    if (heat_operation && base_in->parameters.flow.surface_tension
                              .temperature_dependent_surface_tension_coefficient != 0.0)
      surface_tension_operation->register_temperature_and_normal_vector(
        temp_dof_idx,
        normal_dof_idx,
        &heat_operation->get_temperature(),
        &level_set_operation->get_normal_vector());


    // create recoil pressure operation
    if (base_in->parameters.evapor.recoil.enable)
      {
        recoil_pressure_operation = std::make_shared<Evaporation::RecoilPressureOperation<dim>>(
          *scratch_data,
          base_in->parameters,
          flow_operation->get_dof_handler_idx_velocity(),
          flow_operation->get_quad_idx_velocity(),
          flow_operation->get_dof_handler_idx_pressure(),
          ls_hanging_nodes_dof_idx,
          (base_in->parameters.evapor.recoil.interface_distributed_flux_type ==
           Evaporation::RegularizedRecoilPressureTemperatureEvaluationType::interface_value) ?
            temp_hanging_nodes_dof_idx :
            temp_dof_idx);

        if (base_in->parameters.evapor.recoil.interface_distributed_flux_type ==
            Evaporation::RegularizedRecoilPressureTemperatureEvaluationType::interface_value)
          scratch_data->create_remote_point_evaluation(temp_hanging_nodes_dof_idx);
      }
    /*
     *    initialize the evaporation class
     */
    if (base_in->parameters.evapor.evaporative_cooling.enable ||
        base_in->parameters.evapor.evaporative_dilation_rate.enable)
      {
        evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim>>(
          *scratch_data,
          level_set_operation->get_level_set_as_heaviside(),
          level_set_operation->get_normal_vector(),
          base_in->parameters.evapor,
          base_in->parameters.material,
          normal_dof_idx,
          evapor_vel_dof_idx,
          evapor_mass_flux_dof_idx,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx);

        if (base_in->parameters.evapor.interface_temperature_evaluation_type ==
            Evaporation::EvaporativeMassFluxTemperatureEvaluationType::interface_value)
          scratch_data->create_remote_point_evaluation(evapor_mass_flux_dof_idx);
        if ((base_in->parameters.evapor.formulation_source_term_level_set ==
               Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local ||
             base_in->parameters.evapor.formulation_source_term_level_set ==
               Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp ||
             base_in->parameters.evapor.formulation_source_term_level_set ==
               Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy))
          scratch_data->create_remote_point_evaluation(vel_dof_idx);
        /*
         * register temperature field
         */
        evaporation_operation->reinit(&heat_operation->get_temperature(),
                                      level_set_operation->get_distance_to_level_set(),
                                      base_in->parameters.evapor.recoil,
                                      base_in->parameters.ls.nearest_point,
                                      base_in->parameters.ls.reinit,
                                      temp_dof_idx);

        if (base_in->parameters.evapor.evaporative_dilation_rate.enable &&
            base_in->parameters.evapor.evaporative_dilation_rate.model ==
              Evaporation::InterfaceFluxType::sharp)
          evaporation_operation->register_surface_mesh(
            level_set_operation->get_surface_mesh_info());

          /*
           *  Create a modified viscous stress-strain relation in case of an existing evaporation
           * mass source term, such that div(u)!=0. This material law will be only evaluated if the
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
        if (base_in->parameters.evapor.evaporative_cooling.enable)
          {
            if (base_in->parameters.evapor.evaporative_cooling.enable &&
                base_in->parameters.evapor.evaporative_cooling.model ==
                  Evaporation::EvaporCoolingInterfaceFluxType::sharp)
              heat_operation->register_surface_mesh(level_set_operation->get_surface_mesh_info());

            heat_operation->register_evaporative_mass_flux(
              &evaporation_operation->get_evaporative_mass_flux(),
              evapor_mass_flux_dof_idx,
              base_in->parameters.material.latent_heat_of_evaporation,
              base_in->parameters.evapor.evaporative_cooling);
          }
      }
    /*
     *    initialize the melt pool operation class
     */
    if (problem_specific_parameters.do_solidification)
      {
        melt_front_propagation = std::make_shared<MeltPool::MeltFrontPropagation<dim>>(
          *scratch_data,
          base_in->parameters,
          ls_hanging_nodes_dof_idx,
          heat_operation->get_temperature(),
          reinit_dof_idx,
          reinit_no_solid_dof_idx,
          flow_operation->get_dof_handler_idx_velocity(),
          flow_vel_no_solid_dof_idx,
          temp_hanging_nodes_dof_idx);
        /*
         * Register solid fraction in surface tension
         */
        if (base_in->parameters.flow.surface_tension.zero_surface_tension_in_solid)
          surface_tension_operation->register_solid_fraction(temp_hanging_nodes_dof_idx,
                                                             &melt_front_propagation->get_solid());
        /*
         *    initialize the darcy damping operation class
         */
        if (base_in->parameters.flow.darcy_damping.mushy_zone_morphology > 0.0)
          {
            AssertThrow(heat_operation,
                        ExcMessage("Heat operation needs to be set up "
                                   "for solidification via Darcy damping."));
            darcy_operation = std::make_shared<Flow::DarcyDampingOperation<dim>>(
              base_in->parameters.flow.darcy_damping,
              *scratch_data,
              flow_operation->get_dof_handler_idx_velocity(),
              flow_operation->get_quad_idx_velocity(),
              temp_hanging_nodes_dof_idx);
          }
      }

    if (problem_specific_parameters.do_solidification)
      AssertThrow(
        melt_front_propagation,
        ExcMessage("If solidifcation is enabled the melt pool operation must be initialized! Check "
                   "if the parameter >>>do soldification<<< is set to true. Abort..."));

    /*
     *  set initial conditions of all operations
     */
    set_initial_condition(base_in);
    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(vel_dof_idx),
                                           base_in->parameters.output,
                                           base_in->parameters.time_stepping,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(vel_dof_idx),
                                           scratch_data->get_pcout(1));
    output_interface_velocity =
      (evaporation_operation && base_in->parameters.evapor.evaporative_dilation_rate.enable &&
       (base_in->parameters.evapor.formulation_source_term_level_set ==
          Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local ||
        base_in->parameters.evapor.formulation_source_term_level_set ==
          Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp ||
        base_in->parameters.evapor.formulation_source_term_level_set ==
          Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy));
    /*
     *  initialize profiling
     */
    if (base_in->parameters.profiling.enable)
      profiling_monitor =
        std::make_unique<Profiling::ProfilingMonitor<double>>(base_in->parameters.profiling,
                                                              *time_iterator);
    /*
     *  initialize restart
     */
    if (base_in->parameters.restart.load >= 0 || base_in->parameters.restart.save >= 0)
      {
        restart_monitor =
          std::make_shared<Restart::RestartMonitor<double>>(base_in->parameters.restart,
                                                            *time_iterator);
      }
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
          if (melt_front_propagation)
            str << " solid.size " << melt_front_propagation->get_solid().size();

          Journal::print_line(scratch_data->get_pcout(), str.str(), "melt_pool_problem");
          refine_mesh(base_in);
          /*
           *  set initial conditions after initial AMR
           */
          set_initial_condition(base_in);
        }
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

    // set inflow outflow BC
    level_set_operation->set_inflow_outflow_bc(base_in->get_bc("level_set")->inflow_outflow_bc);

    // set initial conditions of the temperature field
    if (heat_operation)
      {
        if (laser_operation &&
            base_in->parameters.laser.model == Heat::LaserModelType::analytical_temperature)
          Heat::LaserAnalyticalTemperatureField<dim>::compute_temperature_field(
            *scratch_data,
            base_in->parameters.material,
            base_in->parameters.laser,
            laser_operation->get_laser_power(),
            laser_operation->get_laser_position(),
            heat_operation->get_temperature(),
            level_set_operation->get_level_set_as_heaviside(),
            temp_dof_idx);
        else if
          // constant evaporative mass flux --> no need to set initial condition
          (!(evaporation_operation && base_in->parameters.evapor.evaporative_mass_flux_model ==
                                        Evaporation::EvaporationModelType::analytical))
          heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"),
                                                base_in->parameters.time_stepping.start_time);
      }
    /*
     * set initial condition of the melt pool class
     */
    if (melt_front_propagation)
      {
        melt_front_propagation->set_initial_condition(
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
    if (darcy_operation)
      darcy_operation->reinit();

    set_phase_dependent_parameters_flow(base_in->parameters);

    // compute the evaporative mass flux from the initial temperature field
    if (evaporation_operation)
      {
        // E.g. if a spatially constant evaporative mass flux is given as an analytical function,
        // the time is needed to evaluate the function.
        evaporation_operation->set_time(time_iterator->get_current_time());

        evaporation_operation->compute_evaporative_mass_flux();
      }

    if (evaporation_operation && (base_in->parameters.evapor.evaporative_cooling.model ==
                                    Evaporation::EvaporCoolingInterfaceFluxType::sharp ||
                                  base_in->parameters.evapor.evaporative_dilation_rate.model ==
                                    Evaporation::InterfaceFluxType::sharp))
      level_set_operation->update_surface_mesh();
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                         const bool                           do_reinit)
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(base_in->parameters.ls.fe, dof_handler_ls);

    if (problem_specific_parameters.do_heat_transfer)
      FiniteElementUtils::distribute_dofs<dim, 1>(base_in->parameters.heat.fe, dof_handler_heat);

    if (laser_operation)
      laser_operation->distribute_dofs(base_in->parameters.base.fe);

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
    if (problem_specific_parameters.do_heat_transfer)
      base_in->attach_boundary_condition("heat_transfer");

    MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_BC_into_DBC<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("level_set"),
      base_in->get_periodic_bc(),
      ls_dof_idx,
      ls_hanging_nodes_dof_idx);
    MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_BC_into_DBC<dim>(
      *scratch_data,
      base_in->get_dirichlet_bc("level_set"),
      base_in->get_periodic_bc(),
      reinit_dof_idx,
      ls_hanging_nodes_dof_idx,
      false /*set inhomogeneities to zero*/);

    // additional reinitialization dirichlet bc
    if (base_in->get_bc("reinitialization"))
      MeltPoolDG::Constraints::fill_DBC<dim>(
        *scratch_data, base_in->get_dirichlet_bc("reinitialization"), reinit_dof_idx, true, true);
    reinit_no_solid_constraints_dirichlet.copy_from(reinit_constraints_dirichlet);

    if (problem_specific_parameters.do_heat_transfer)
      MeltPoolDG::Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_BC_into_DBC<dim>(
        *scratch_data,
        base_in->get_dirichlet_bc("heat_transfer"),
        base_in->get_periodic_bc(),
        temp_dof_idx,
        temp_hanging_nodes_dof_idx);

    if (laser_operation)
      laser_operation->setup_constraints();

    // TODO: add function to each operation to check the requirements on ScratchData
    scratch_data->build(/*enable_boundary_face_loops*/ problem_specific_parameters.do_heat_transfer,
                        /*enable_inner_face_loops*/ (
                          base_in->parameters.laser.model ==
                            Heat::LaserModelType::interface_projection_sharp_conforming ||
                          base_in->parameters.evapor.evaporative_cooling.model ==
                            Evaporation::EvaporCoolingInterfaceFluxType::sharp_conforming));

    if (do_reinit)
      {
        {
          ScopedName sc("mp::cells");
          CellMonitor::add_info(sc,
                                scratch_data->get_triangulation().n_global_active_cells(),
                                scratch_data->get_min_cell_size(),
                                scratch_data->get_max_cell_size());
        }

        level_set_operation->reinit();

        if (evaporation_operation)
          evaporation_operation->reinit();
        if (melt_front_propagation)
          melt_front_propagation->reinit();
        if (heat_operation)
          heat_operation->reinit();
        if (laser_operation)
          laser_operation->reinit();
        if (darcy_operation)
          darcy_operation->reinit();
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
    {
      ScopedName sc("mp::cells");
      CellMonitor::add_info(sc,
                            scratch_data->get_triangulation().n_global_active_cells(),
                            scratch_data->get_min_cell_size(),
                            scratch_data->get_max_cell_size());
    }
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::set_phase_dependent_parameters_flow(const Parameters<double> &parameters)
  {
    // compute damping coefficients at the quadrature points of the fluid
    // solver
    if (darcy_operation)
      {
        darcy_operation->set_darcy_damping_at_q(*material,
                                                level_set_operation->get_level_set_as_heaviside(),
                                                heat_operation->get_temperature(),
                                                ls_hanging_nodes_dof_idx,
                                                temp_dof_idx);
      }

    // compute density and viscosity at the quadrature points.
    const bool ls_update_ghosts =
      !level_set_operation->get_level_set_as_heaviside().has_ghost_elements();
    if (ls_update_ghosts)
      level_set_operation->get_level_set_as_heaviside().update_ghost_values();

    bool temp_update_ghosts = false;
    if (material->has_dependency(Material<double>::FieldType::temperature) && heat_operation)
      {
        temp_update_ghosts = !heat_operation->get_temperature().has_ghost_elements();

        if (temp_update_ghosts)
          heat_operation->get_temperature().update_ghost_values();
      }

    double dummy;
    scratch_data->get_matrix_free().template cell_loop<double, VectorType>(
      [&](const auto &matrix_free, auto &, const auto &ls_as_heaviside, auto macro_cells) {
        FECellIntegrator<dim, 1, double> ls_values(matrix_free,
                                                   ls_hanging_nodes_dof_idx,
                                                   flow_operation->get_quad_idx_velocity());

        FECellIntegrator<dim, 1, double> temp_values(matrix_free,
                                                     temp_dof_idx,
                                                     flow_operation->get_quad_idx_velocity());

        for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            ls_values.reinit(cell);
            ls_values.read_dof_values_plain(ls_as_heaviside);
            ls_values.evaluate(EvaluationFlags::values);

            if (heat_operation &&
                material->has_dependency(Material<double>::FieldType::temperature))
              {
                temp_values.reinit(cell);
                temp_values.read_dof_values_plain(heat_operation->get_temperature());
                temp_values.evaluate(EvaluationFlags::values);
              }

            for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
              {
                auto material_values =
                  material->template compute_parameters<VectorizedArray<double>>(
                    ls_values,
                    temp_values,
                    MaterialUpdateFlags::density | MaterialUpdateFlags::dynamic_viscosity,
                    q);

                // set density and viscosity of the fluid solver
                flow_operation->get_density(cell, q)   = material_values.density;
                flow_operation->get_viscosity(cell, q) = material_values.dynamic_viscosity;

                // set damping coefficient of the fluid solver
                if (darcy_operation && (parameters.flow.darcy_damping.formulation ==
                                        DarcyDampingFormulation::implicit_formulation))
                  flow_operation->get_damping(cell, q) = darcy_operation->get_damping(cell, q);
              }
          }
      },
      dummy,
      level_set_operation->get_level_set_as_heaviside());

#ifdef MELT_POOL_DG_WITH_ADAFLO
    dynamic_cast<Flow::AdafloWrapper<dim> *>(flow_operation.get())
      ->set_face_average_density_augmented_taylor_hood(
        *material,
        level_set_operation->get_level_set_as_heaviside(),
        ls_hanging_nodes_dof_idx,
        ((heat_operation) ? &heat_operation->get_temperature() : nullptr),
        temp_dof_idx);
#endif
    if (ls_update_ghosts)
      level_set_operation->get_level_set_as_heaviside().zero_out_ghost_values();
    if (temp_update_ghosts)
      heat_operation->get_temperature().zero_out_ghost_values();
  }



  template <int dim>
  void
  MeltPoolProblem<dim>::compute_gravity_force(VectorType  &vec,
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
  MeltPoolProblem<dim>::compute_interface_velocity(
    const LevelSet::LevelSetData<double>       &ls_data,
    const Evaporation::EvaporationData<double> &evapor_data)
  {
    // TODO: remove; could not be removed since during
    // flow_operation->init_time_advance() an inconsistency in the DoF vectors is
    // introduced
    scratch_data->initialize_dof_vector(interface_velocity, vel_dof_idx);
    interface_velocity.copy_locally_owned_data_from(flow_operation->get_velocity());

    const bool do_sharp_velocity =
      (evapor_data.formulation_source_term_level_set ==
         Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp ||
       evapor_data.formulation_source_term_level_set ==
         Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy);

    if (not do_sharp_velocity && not evapor_data.evaporative_dilation_rate.enable)
      return;

    if (evaporation_operation && evapor_data.evaporative_dilation_rate.enable)
      {
        ScopedName         sc("evaporation::level_set_source_term");
        TimerOutput::Scope scope(scratch_data->get_timer(), sc);
        switch (evapor_data.formulation_source_term_level_set)
          {
            default:
            case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp:
            case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy:
              case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local: {
                // Option 1: compute modified advection velocity due to evaporation
                if (problem_specific_parameters.do_extrapolate_coupling_terms)
                  {
                    level_set_operation->update_normal_vector();
                  }

                evaporation_operation->compute_evaporation_velocity();
                interface_velocity += evaporation_operation->get_velocity();

                if (do_sharp_velocity)
                  this->compute_interface_velocity_sharp(ls_data, evapor_data);

                break;
              }
            case Evaporation::EvaporationLevelSetSourceTermType::rhs:
              // Option 2: use source term as rhs in the level set equation
              scratch_data->initialize_dof_vector(level_set_rhs, ls_dof_idx);
              evaporation_operation->compute_level_set_source_term(
                level_set_rhs, ls_dof_idx, level_set_operation->get_level_set(), pressure_dof_idx);
              level_set_operation->set_level_set_user_rhs(level_set_rhs);
              break;
          }
      }
    else // evaporative_dilation_rate.enable = false
      {
        if (do_sharp_velocity)
          this->compute_interface_velocity_sharp(ls_data, evapor_data);
      }

    // distribute hanging node constraints
    flow_operation->get_hanging_node_constraints_velocity().distribute(interface_velocity);
  }



  template <int dim>
  void
  MeltPoolProblem<dim>::compute_interface_velocity_sharp(
    const LevelSet::LevelSetData<double>       &ls_data,
    const Evaporation::EvaporationData<double> &evapor_data)
  {
    VectorType interface_velocity_interface;
    scratch_data->initialize_dof_vector(interface_velocity_interface, vel_dof_idx);

    LevelSet::NearestPointData<double> nearest_point_data = ls_data.nearest_point;

    // compute isocontour from which the velocity should be extrapolated
    switch (evapor_data.formulation_source_term_level_set)
      {
          case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp: {
            // in the default case, the velocity is extended from the zero-isosurface of the signed
            // distance function.
            nearest_point_data.isocontour = 0;
            break;
          }
          case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_sharp_heavy: {
            // in case the velocity should be extended from the liquid end of the interface region,
            // use the isocontour of the signed distance function d=3ε where the smoothed heaviside
            // function attains 1.
            nearest_point_data.isocontour =
              ls_data.reinit.compute_interface_thickness_parameter_epsilon(
                scratch_data->get_min_cell_size(ls_dof_idx)) *
              3.;
            break;
          }
        case Evaporation::EvaporationLevelSetSourceTermType::interface_velocity_local:
        case Evaporation::EvaporationLevelSetSourceTermType::rhs:
          DEAL_II_ASSERT_UNREACHABLE();
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    LevelSet::Tools::NearestPoint<dim> nearest_point(
      scratch_data->get_mapping(),
      scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx),
      level_set_operation->get_distance_to_level_set(),
      level_set_operation->get_normal_vector(),
      scratch_data->get_remote_point_evaluation(vel_dof_idx),
      nearest_point_data,
      scratch_data->get_timer());

    nearest_point.reinit(scratch_data->get_dof_handler(vel_dof_idx));

    nearest_point.template fill_dof_vector_with_point_values<dim>(
      interface_velocity_interface, scratch_data->get_dof_handler(vel_dof_idx), interface_velocity);
    interface_velocity.swap(interface_velocity_interface);
  }



  template <int dim>
  void
  MeltPoolProblem<dim>::output_results(
    std::shared_ptr<SimulationBase<dim>> base_in,
    const bool                           force_output,
    const OutputNotConvergedOperation    output_not_converged_operation)
  {
    const unsigned int n_time_step  = time_iterator->get_current_time_step_number();
    const double       current_time = time_iterator->get_current_time();

    if (!post_processor->is_output_timestep(n_time_step, current_time) && !force_output &&
        !base_in->parameters.output.do_user_defined_postprocessing)
      return;

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(),
                                         current_time,
                                         base_in->parameters.output.output_variables);
    attach_output_vectors(generic_data_out);

    switch (output_not_converged_operation)
      {
          case OutputNotConvergedOperation::navier_stokes: {
            flow_operation->attach_output_vectors_failed_step(generic_data_out);
            break;
          }
          case OutputNotConvergedOperation::heat_transfer: {
            heat_operation->attach_output_vectors_failed_step(generic_data_out);
            break;
          }
        case OutputNotConvergedOperation::none:
          break;
      }

    // user-defined postprocessing
    if (base_in->parameters.output.do_user_defined_postprocessing)
      base_in->do_postprocessing(generic_data_out);

    // postprocessing
    {
      ScopedName         sc("process");
      TimerOutput::Scope scope(scratch_data->get_timer(), sc);

      post_processor->process(n_time_step,
                              generic_data_out,
                              current_time,
                              force_output,
                              output_not_converged_operation != OutputNotConvergedOperation::none);
    }
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::finalize(std::shared_ptr<SimulationBase<dim>> base_in,
                                 const OutputNotConvergedOperation    output_no_converged_operation)
  {
    output_results(base_in, true /* force_output */, output_no_converged_operation);

    //... always print timing statistics
    if (profiling_monitor)
      profiling_monitor->print(scratch_data->get_pcout(),
                               scratch_data->get_timer(),
                               scratch_data->get_mpi_comm());
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    level_set_operation->attach_output_vectors(data_out);

    flow_operation->attach_output_vectors(data_out);

    if (melt_front_propagation)
      melt_front_propagation->attach_output_vectors(data_out);

    if (evaporation_operation)
      evaporation_operation->attach_output_vectors(data_out);
    if (heat_operation)
      heat_operation->attach_output_vectors(data_out);
    if (laser_operation)
      laser_operation->attach_output_vectors(data_out);
    if (darcy_operation)
      darcy_operation->attach_output_vectors(data_out);

    if (output_interface_velocity)
      {
        /*
         *  interface velocity
         */
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          vector_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(scratch_data->get_dof_handler(vel_dof_idx),
                                 interface_velocity,
                                 std::vector<std::string>(dim, "interface_velocity"),
                                 vector_component_interpretation);
      }
  }

  template <int dim>
  bool
  MeltPoolProblem<dim>::mark_cells_for_refinement(std::shared_ptr<SimulationBase<dim>> base_in,
                                                  Triangulation<dim>                  &tria)
  {
    const auto &amr_data = base_in->parameters.amr;

    const bool ls_update_ghosts = !level_set_operation->get_level_set().has_ghost_elements();
    if (ls_update_ghosts)
      level_set_operation->get_level_set().update_ghost_values();

    const bool normal_update_ghosts =
      !level_set_operation->get_normal_vector().has_ghost_elements();
    if (normal_update_ghosts)
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

        FEValues<dim> ls_values(scratch_data->get_mapping(),
                                scratch_data->get_fe(ls_dof_idx),
                                quadrature,
                                update_values);
        FEValues<dim> vel_values(scratch_data->get_mapping(),
                                 scratch_data->get_fe(vel_dof_idx),
                                 quadrature,
                                 update_values);

        // solution variables
        std::vector<std::vector<double>> ls_gradients(dim, std::vector<double>(quadrature.size()));
        const double                     diffusion_length =
          base_in->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
            scratch_data->get_min_cell_size() / base_in->parameters.ls.get_n_subdivisions());

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
            AssertThrow(base_in->parameters.ls.get_n_subdivisions() <= 1,
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
            if (problem_specific_parameters.do_solidification)
              {
                // 3a) copy the solution
                locally_relevant_solution.reinit(
                  scratch_data->get_partitioner(temp_hanging_nodes_dof_idx));
                locally_relevant_solution.copy_locally_owned_data_from(
                  melt_front_propagation->get_solid());
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

            FEValues<dim>               ls_values(scratch_data->get_mapping(),
                                    scratch_data->get_fe(ls_dof_idx),
                                    quadrature,
                                    update_values);
            FEValues<dim>               vel_values(scratch_data->get_mapping(),
                                     scratch_data->get_fe(vel_dof_idx),
                                     quadrature,
                                     update_values);
            std::vector<Tensor<1, dim>> vel_vals(quadrature.size());

            // solution variables
            std::vector<std::vector<double>> ls_gradients(dim,
                                                          std::vector<double>(quadrature.size()));
            const double                     diffusion_length =
              base_in->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
                scratch_data->get_min_cell_size() / base_in->parameters.ls.get_n_subdivisions());

            std::vector<double> ls_vals(quadrature.size());

            const FEValuesExtractors::Vector velocity(0);

            typename DoFHandler<dim>::active_cell_iterator vel_cell =
              scratch_data->get_dof_handler(vel_dof_idx).begin_active();

            const bool vel_update_ghosts = !flow_operation->get_velocity().has_ghost_elements();
            if (vel_update_ghosts)
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

            if (vel_update_ghosts)
              flow_operation->get_velocity().zero_out_ghost_values();
            break;
          }
      }

    if (melt_front_propagation)
      {
        const bool liq_update_ghosts = !melt_front_propagation->get_liquid().has_ghost_elements();
        if (liq_update_ghosts)
          melt_front_propagation->get_liquid().update_ghost_values();

        const bool sol_update_ghosts = !melt_front_propagation->get_solid().has_ghost_elements();
        if (sol_update_ghosts)
          melt_front_propagation->get_solid().update_ghost_values();

        const bool temp_update_ghosts = !heat_operation->get_temperature().has_ghost_elements();
        if (temp_update_ghosts)
          heat_operation->get_temperature().update_ghost_values();

        Vector<double> liq_vals(scratch_data->get_fe(temp_hanging_nodes_dof_idx).n_dofs_per_cell());
        Vector<double> solid_vals(
          scratch_data->get_fe(temp_hanging_nodes_dof_idx).n_dofs_per_cell());
        Vector<double> temp_vals(
          scratch_data->get_fe(temp_hanging_nodes_dof_idx).n_dofs_per_cell());

        for (const auto &cell :
             scratch_data->get_dof_handler(temp_hanging_nodes_dof_idx).active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            cell->get_dof_values(melt_front_propagation->get_liquid(), liq_vals);
            cell->get_dof_values(melt_front_propagation->get_solid(), solid_vals);
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
              else if ((solid_vals[i] > 0.0 || problem_specific_parameters.amr.refine_gas_domain) &&
                       temp_vals[i] >= problem_specific_parameters.amr
                                           .fraction_of_melting_point_refined_in_solid *
                                         base_in->parameters.material.solidus_temperature)
                {
                  cell->clear_coarsen_flag();
                  cell->set_refine_flag();
                  break;
                }
          }
        if (liq_update_ghosts)
          melt_front_propagation->get_liquid().zero_out_ghost_values();
        if (sol_update_ghosts)
          melt_front_propagation->get_solid().zero_out_ghost_values();
        if (temp_update_ghosts)
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

    if (ls_update_ghosts)
      level_set_operation->get_level_set().zero_out_ghost_values();
    if (normal_update_ghosts)
      level_set_operation->get_normal_vector().zero_out_ghost_values();
    return true;
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::attach_vectors(
    std::vector<
      std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>> &data)
  {
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

    if (melt_front_propagation)
      data.emplace_back(&dof_handler_heat, [&](std::vector<VectorType *> &vectors) {
        melt_front_propagation->attach_vectors(vectors); // temperature + solid + liquid
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

    if (laser_operation)
      laser_operation->attach_vectors(data);
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::post()
  {
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
    if (melt_front_propagation)
      melt_front_propagation->distribute_constraints();
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

    if (laser_operation)
      laser_operation->distribute_constraints();
  }

  template <int dim>
  void
  MeltPoolProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement = [&](Triangulation<dim> &tria) -> bool {
      return this->mark_cells_for_refinement(base_in, tria);
    };

    const auto attach_vectors =
      [&](std::vector<std::pair<const DoFHandler<dim> *,
                                std::function<void(std::vector<VectorType *> &)>>> &data) {
        this->attach_vectors(data);
      };

    const auto post = [&]() { this->post(); };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 *base_in->triangulation,
                                 time_iterator->get_current_time_step_number());
  }

  template class MeltPoolProblem<MELT_POOL_DG_DIM>;
} // namespace MeltPoolDG::MeltPool
