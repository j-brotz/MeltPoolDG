#include <meltpooldg/heat/heat_transfer_problem.hpp>
//

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/numerics/error_estimator.h>

#include <meltpooldg/heat/laser_heat_source_gauss.hpp>
#include <meltpooldg/heat/laser_heat_source_gusarov.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  void
  HeatTransferProblem<dim>::run(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    initialize(base_in);

    while (!time_iterator.is_finished())
      {
        const auto dt = time_iterator.get_next_time_increment();
        const auto n  = time_iterator.get_current_time_step_number();

        scratch_data->get_pcout() << "t= " << std::setw(10) << std::left
                                  << time_iterator.get_current_time();

        if (const auto velocity_field_function = base_in->get_velocity_field("heat_transfer"))
          compute_field_vector(velocity, velocity_dof_idx, *velocity_field_function);

        if (base_in->parameters.heat.two_phase)
          compute_field_vector(level_set_as_heaviside,
                               level_set_dof_idx,
                               *base_in->get_initial_condition("prescribed_level_set"));

        if (const auto source_field_function = base_in->get_source_field("heat_transfer"))
          compute_field_vector(heat_operation->get_heat_source(),
                               temp_dof_idx,
                               *source_field_function);
        // TODO: Atm the laser heat source will overwrite the heat source field function if both are
        // given. Instead they should be added.
        if (laser_operation)
          {
            laser_operation->move_laser(dt);
            const auto impact_type = laser_operation->get_laser_impact_type();
            if (impact_type == volumetric)
              laser_heat_source_operation->compute_volumetric_heat_source(
                heat_operation->get_heat_source(),
                *scratch_data,
                temp_dof_idx,
                laser_operation->get_laser_power(),
                laser_operation->get_laser_position(),
                true /* zero_out */);
            else if (impact_type == interface)
              laser_heat_source_operation->compute_interfacial_heat_source(
                heat_operation->get_heat_source(),
                *scratch_data,
                temp_dof_idx,
                laser_operation->get_laser_power(),
                laser_operation->get_laser_position(),
                heat_operation->get_level_set_as_heaviside(),
                level_set_dof_idx,
                true /* zero_out */);
            else
              AssertThrow(false, ExcMessage("Unknown laser impact type! Abort..."));
          }

        heat_operation->solve(dt);

        // ... and output the results to vtk files.
        output_results(n, time_iterator.get_current_time(), base_in);

        if (base_in->parameters.amr.do_amr)
          refine_mesh(base_in);
      }
  }

  template <int dim>
  std::string
  HeatTransferProblem<dim>::get_name()
  {
    return "heat_transfer";
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::initialize(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /*
     *  setup DoFHandler
     */
    dof_handler.reinit(*base_in->triangulation);
    dof_handler_velocity.reinit(*base_in->triangulation);
    dof_handler_level_set.reinit(*base_in->triangulation);
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
    scratch_data->attach_dof_handler(dof_handler_velocity);
    scratch_data->attach_dof_handler(dof_handler_level_set);

    /*
     * attach constraints
     */
    temp_dof_idx = scratch_data->attach_constraint_matrix(temp_constraints);
    temp_hanging_nodes_dof_idx =
      scratch_data->attach_constraint_matrix(temp_hanging_nodes_constraints);
    velocity_dof_idx  = scratch_data->attach_constraint_matrix(velocity_hanging_nodes_constraints);
    level_set_dof_idx = scratch_data->attach_constraint_matrix(level_set_hanging_nodes_constraints);

    /*
     *  create quadrature rule
     */
    if (base_in->parameters.base.do_simplex)
      {
        temp_quad_idx = scratch_data->attach_quadrature(
          QGaussSimplex<dim>(base_in->parameters.base.n_q_points_1d));
      }
    else
      {
        temp_quad_idx =
          scratch_data->attach_quadrature(QGauss<dim>(base_in->parameters.base.n_q_points_1d));
      }

    setup_dof_system(base_in, false);
    /*
     *  initialize the time stepping scheme
     */
    time_iterator.initialize(
      TimeIteratorData<double>{base_in->parameters.heat.time_stepping.start_time,
                               base_in->parameters.heat.time_stepping.end_time,
                               base_in->parameters.heat.time_stepping.time_step_size,
                               base_in->parameters.heat.time_stepping.max_n_steps,
                               false /*cfl_condition-->not supported yet*/});
    /*
     *  set initial conditions of the levelset function
     */
    AssertThrow(
      base_in->get_initial_condition("heat_transfer"),
      ExcMessage(
        "It seems that your SimulationBase object does not contain "
        "a valid initial field function for the temperature field. A shared_ptr to your initial field "
        "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
        "  this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), 'temperature') "));
    /*
     *    set velocity field
     */
    VectorType *velocity_ptr = nullptr;
    if (base_in->parameters.heat.velocity != 0.0)
      {
        compute_field_vector(velocity,
                             velocity_dof_idx,
                             *base_in->get_velocity_field("heat_transfer"));
        velocity_ptr = &velocity;
      }

    /*
     *    set level-set as heaviside field
     */
    VectorType *level_set_as_heaviside_ptr = nullptr;
    if (base_in->parameters.heat.two_phase)
      {
        compute_field_vector(level_set_as_heaviside,
                             level_set_dof_idx,
                             *base_in->get_initial_condition("prescribed_level_set"));
        level_set_as_heaviside_ptr = &level_set_as_heaviside;
      }

    if (base_in->parameters.laser.power > 0.0)
      {
        /*
         *  initialize the laser operation class
         */
        laser_operation = std::make_shared<Heat::LaserOperation<dim>>(*scratch_data,
                                                                      base_in->parameters.laser,
                                                                      base_in->parameters.material);
        laser_operation->set_initial_condition(base_in->parameters.heat.time_stepping.start_time);

        if (base_in->parameters.laser.heat_source_model == "Gusarov")
          {
            laser_heat_source_operation = std::make_shared<Heat::LaserHeatSourceGusarov<dim>>(
              base_in->parameters.laser.gusarov);
          }
        else if (base_in->parameters.laser.heat_source_model == "Gauss")
          {
            laser_heat_source_operation =
              std::make_shared<Heat::LaserHeatSourceGauss<dim>>(base_in->parameters.laser.gauss);
          }
        else
          AssertThrow(false,
                      ExcMessage(
                        "No requested laser model found. Please speficy the "
                        "heat source model in the laser section of the input parameters."));
      }

    /*
     *    initialize the heat operation class
     */
    heat_operation = std::make_shared<HeatTransferOperation<dim>>(base_in->get_bc("heat_transfer"),
                                                                  *scratch_data,
                                                                  base_in->parameters.heat,
                                                                  base_in->parameters.material,
                                                                  temp_dof_idx,
                                                                  temp_hanging_nodes_dof_idx,
                                                                  temp_quad_idx,
                                                                  velocity_dof_idx,
                                                                  velocity_ptr,
                                                                  level_set_dof_idx,
                                                                  level_set_as_heaviside_ptr);

    heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"));

    /*
     *  initialize postprocessor
     */
    post_processor =
      std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(temp_dof_idx),
                                           base_in->parameters.paraview,
                                           scratch_data->get_mapping(),
                                           scratch_data->get_triangulation(temp_dof_idx));
    /*
     *    Do initial refinement steps if requested
     */
    if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
      for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
        {
          scratch_data->get_pcout()
            << " T.size " << heat_operation->get_temperature().size() << std::endl;
          refine_mesh(base_in);
          /*
           *  set initial conditions after initial AMR
           */
          heat_operation->set_initial_condition(*base_in->get_initial_condition("heat_transfer"));
        }
    /*
     *  output results of initialization
     */
    output_results(0, base_in->parameters.heat.time_stepping.start_time, base_in);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::compute_field_vector(VectorType &       vector,
                                                 const unsigned int dof_idx,
                                                 Function<dim> &    field_function)
  {
    scratch_data->initialize_dof_vector(vector, dof_idx);
    /*
     *  set the current time to the advection field function
     */
    field_function.set_time(time_iterator.get_current_time());
    /*
     *  interpolate the values of the advection velocity
     */
    dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(dof_idx),
                                     field_function,
                                     vector);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in,
                                             const bool                           do_reinit)
  {
    if (base_in->parameters.base.do_simplex)
      {
        dof_handler.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
        dof_handler_velocity.distribute_dofs(
          FESystem<dim>(FE_SimplexP<dim>(base_in->parameters.base.degree), dim));
        dof_handler_level_set.distribute_dofs(FE_SimplexP<dim>(base_in->parameters.base.degree));
      }
    else
      {
        dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
        dof_handler_velocity.distribute_dofs(
          FESystem<dim>(FE_Q<dim>(base_in->parameters.base.degree), dim));
        dof_handler_level_set.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
      }
    /*
     *  create partitioning
     */
    scratch_data->create_partitioning();
    /*
     *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
     *  dirichlet constraints are supported)
     */
    temp_hanging_nodes_constraints.clear();
    temp_hanging_nodes_constraints.reinit(
      scratch_data->get_locally_relevant_dofs(temp_hanging_nodes_dof_idx));
    DoFTools::make_hanging_node_constraints(dof_handler, temp_hanging_nodes_constraints);
    temp_hanging_nodes_constraints.close();

    temp_constraints.clear();
    temp_constraints.reinit(scratch_data->get_locally_relevant_dofs(temp_dof_idx));
    if (base_in->get_bc("heat_transfer") && !base_in->get_dirichlet_bc("heat_transfer").empty())
      {
        for (const auto &bc : base_in->get_dirichlet_bc(
               "heat_transfer")) // @todo: add name of bc at a more central place
          {
            dealii::VectorTools::interpolate_boundary_values(
              scratch_data->get_mapping(), dof_handler, bc.first, *bc.second, temp_constraints);
          }
      }
    temp_constraints.close();
    temp_constraints.merge(temp_constraints,
                           AffineConstraints<double>::MergeConflictBehavior::right_object_wins);

    velocity_hanging_nodes_constraints.clear();
    velocity_hanging_nodes_constraints.reinit(
      scratch_data->get_locally_relevant_dofs(velocity_dof_idx));
    DoFTools::make_hanging_node_constraints(dof_handler_velocity,
                                            velocity_hanging_nodes_constraints);
    velocity_hanging_nodes_constraints.close();

    level_set_hanging_nodes_constraints.clear();
    level_set_hanging_nodes_constraints.reinit(
      scratch_data->get_locally_relevant_dofs(level_set_dof_idx));
    DoFTools::make_hanging_node_constraints(dof_handler_level_set,
                                            level_set_hanging_nodes_constraints);
    level_set_hanging_nodes_constraints.close();

    scratch_data->build();

    if (do_reinit)
      heat_operation->reinit();
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::output_results(const unsigned int                   n_time_step,
                                           const double                         time,
                                           std::shared_ptr<SimulationBase<dim>> base_in)
  {
    /**
     * collect all relevant output data
     */
    const auto attach_output_vectors = [&](GenericDataOut<dim> &data_out) {
      heat_operation->attach_output_vectors(data_out);
    };

    GenericDataOut<dim> generic_data_out(scratch_data->get_mapping(), time);
    attach_output_vectors(generic_data_out);

    // user-defined postprocessing
    base_in->do_postprocessing(generic_data_out);

    // paraview postprocessing
    post_processor->process(n_time_step, attach_output_vectors, time);
  }

  template <int dim>
  void
  HeatTransferProblem<dim>::refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    const auto mark_cells_for_refinement =
      [&](parallel::distributed::Triangulation<dim> &tria) -> bool {
      Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

      VectorType locally_relevant_solution;
      locally_relevant_solution.reinit(scratch_data->get_partitioner(temp_dof_idx));
      locally_relevant_solution.copy_locally_owned_data_from(heat_operation->get_temperature());
      scratch_data->get_constraint(temp_dof_idx).distribute(locally_relevant_solution);
      locally_relevant_solution.update_ghost_values();

      KellyErrorEstimator<dim>::estimate(scratch_data->get_mapping(),
                                         scratch_data->get_dof_handler(temp_dof_idx),
                                         scratch_data->get_face_quadrature(temp_quad_idx),
                                         {}, // neumann bc
                                         locally_relevant_solution,
                                         estimated_error_per_cell);
      auto vec =
        Utilities::MPI::gather(scratch_data->get_mpi_comm(), estimated_error_per_cell.l2_norm());

      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        tria,
        estimated_error_per_cell,
        base_in->parameters.amr.upper_perc_to_refine,
        base_in->parameters.amr.lower_perc_to_coarsen);

      return true;
    };

    const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
      heat_operation->attach_vectors(vectors);
    };

    const auto post = [&]() { heat_operation->distribute_constraints(); };

    const auto setup_dof_system = [&]() { this->setup_dof_system(base_in, true); };

    refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                 attach_vectors,
                                 post,
                                 setup_dof_system,
                                 base_in->parameters.amr,
                                 dof_handler,
                                 time_iterator.get_current_time_step_number());
  }

  template class HeatTransferProblem<1>;
  template class HeatTransferProblem<2>;
  template class HeatTransferProblem<3>;
} // namespace MeltPoolDG::Heat
