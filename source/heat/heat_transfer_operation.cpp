#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/newton_raphson_solver.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  HeatTransferOperation<dim>::HeatTransferOperation(
    std::shared_ptr<BoundaryConditions<dim>> bc_data_in,
    const ScratchData<dim> &                 scratch_data_in,
    const HeatData<double> &                 heat_data_in,
    const Material<double> &                 material,
    const TimeIterator<double> &             time_iterator,
    const unsigned int                       temp_dof_idx_in,
    const unsigned int                       temp_hanging_nodes_dof_idx_in,
    const unsigned int                       temp_quad_idx_in,
    const unsigned int                       vel_dof_idx_in,
    VectorType *                             velocity_in,
    const unsigned int                       ls_dof_idx_in,
    VectorType *                             level_set_as_heaviside_in)
    : scratch_data(scratch_data_in)
    , bc_data(bc_data_in)
    , heat_data(heat_data_in)
    , time_iterator(time_iterator)
    , temp_dof_idx(temp_dof_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , temp_quad_idx(temp_quad_idx_in)
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , ls_dof_idx(ls_dof_idx_in)
    , level_set_as_heaviside(level_set_as_heaviside_in)
  {
    heat_operator = std::make_shared<HeatTransferOperator<dim>>(bc_data,
                                                                scratch_data,
                                                                heat_data,
                                                                material,
                                                                temp_dof_idx,
                                                                temp_quad_idx,
                                                                temp_hanging_nodes_dof_idx,
                                                                temperature,
                                                                temperature_old,
                                                                heat_source,
                                                                vel_dof_idx,
                                                                velocity,
                                                                ls_dof_idx,
                                                                level_set_as_heaviside);


    /*
     * setup preconditioner for matrix-free computation
     */
    heat_transfer_preconditioner = std::make_shared<HeatTransferPreconditionerMatrixFree<dim>>(
      scratch_data, temp_dof_idx, heat_data.linear_solver.preconditioner_type, heat_operator);

    reinit();
  }



  template <int dim>
  void
  HeatTransferOperation<dim>::register_evaporative_mass_flux(
    VectorType *       evaporative_mass_flux_in,
    const unsigned int evapor_mass_flux_dof_idx_in,
    const double       latent_heat_of_evaporation,
    const bool         do_phenomenological_recoil_pressure)
  {
    heat_operator->register_evaporative_mass_flux(evaporative_mass_flux_in,
                                                  evapor_mass_flux_dof_idx_in,
                                                  latent_heat_of_evaporation,
                                                  do_phenomenological_recoil_pressure);
  }


  template <int dim>
  void
  HeatTransferOperation<dim>::register_surface_mesh(
    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<double> /*weights*/
                                 >> &surface_mesh_info_in)
  {
    heat_operator->register_surface_mesh(surface_mesh_info_in);
  }


  template <int dim>
  void
  HeatTransferOperation<dim>::set_initial_condition(
    const Function<dim> &initial_field_function_temperature,
    const double         start_time)
  {
    reinit();

    if (heat_data.enable_time_dependent_bc)
      bc_data->set_time(start_time);

    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(temp_dof_idx),
                                     initial_field_function_temperature,
                                     temperature);

    if (heat_data.enable_time_dependent_bc)
      MeltPoolDG::UtilityFunctions::setup_and_merge_constraints<dim>(const_cast<ScratchData<dim> &>(
                                                                       scratch_data),
                                                                     bc_data->dirichlet_bc,
                                                                     temp_dof_idx,
                                                                     temp_hanging_nodes_dof_idx);

    scratch_data.get_constraint(temp_dof_idx).distribute(temperature);
    temperature_old.copy_locally_owned_data_from(temperature);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::reinit()
  {
    scratch_data.initialize_dof_vector(temperature, temp_dof_idx);
    scratch_data.initialize_dof_vector(temperature_old, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(user_rhs, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(heat_source, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(temperature_interface, temp_hanging_nodes_dof_idx);
    /*
     * setup sparsity pattern of system matrix only if the latter is
     * needed for computing the preconditioner
     */
    heat_transfer_preconditioner->reinit();
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::init_time_advance()
  {
    heat_operator->reset_time_increment(time_iterator.get_current_time_increment());

    if (heat_data.enable_time_dependent_bc)
      {
        bc_data->set_time(time_iterator.get_current_time());
        MeltPoolDG::UtilityFunctions::setup_and_merge_constraints<dim>(
          const_cast<ScratchData<dim> &>(scratch_data),
          bc_data->dirichlet_bc,
          temp_dof_idx,
          temp_hanging_nodes_dof_idx);
      }

    if (heat_data.predictor == PredictorType::linear_extrapolation)
      {
        VectorType temperature_extrapolated;
        scratch_data.initialize_dof_vector(temperature_extrapolated, temp_dof_idx);

        UtilityFunctions::compute_linear_predictor(temperature,
                                                   temperature_old,
                                                   temperature_extrapolated,
                                                   time_iterator.get_current_time_increment(),
                                                   time_iterator.get_old_time_increment());

        temperature_old.copy_locally_owned_data_from(temperature);
        temperature.copy_locally_owned_data_from(temperature_extrapolated);
      }
    else if (heat_data.predictor == PredictorType::none)
      AssertThrow(false, ExcNotImplemented());

    // apply hanging node constraints to predictor
    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(temperature);

    ready_for_time_advance = true;
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::solve()
  {
    if (!ready_for_time_advance)
      init_time_advance();

    if (!heat_data.linear_solver.do_matrix_free)
      AssertThrow(false, ExcNotImplemented());

    // setup preconditioner
    heat_operator->update_ghost_values();
    switch (heat_data.linear_solver.preconditioner_type)
      {
        case PreconditionerType::Diagonal:
          case PreconditionerType::DiagonalReduced: {
            diag_preconditioner = heat_transfer_preconditioner->compute_diagonal_preconditioner();
            break;
          }
        case PreconditionerType::Identity:
        case PreconditionerType::AMG:
        case PreconditionerType::AMGReduced:
        case PreconditionerType::ILU:
          case PreconditionerType::ILUReduced: {
            trilinos_preconditioner =
              heat_transfer_preconditioner->compute_trilinos_preconditioner();
            break;
          }
          default: {
            AssertThrow(false, ExcNotImplemented());
          }
      }

    const auto create_rhs = [&](VectorType &rhs) {
      // solely homogeneous dirichlet bc are distributed for the
      // corrected temperature field in the newton solver
      heat_operator->update_ghost_values();
      rhs.copy_locally_owned_data_from(user_rhs);
      heat_operator->create_rhs(rhs, temperature_old);
    };


    const auto solve_linear_system = [&](VectorType &      solution_update,
                                         const VectorType &rhs) -> int {
      if (diag_preconditioner)
        return LinearSolver::solve<VectorType, OperatorBase<dim, double>>(
          *heat_operator, solution_update, rhs, heat_data.linear_solver, *diag_preconditioner);
      else if (trilinos_preconditioner)
        return LinearSolver::solve<VectorType, OperatorBase<dim, double>>(
          *heat_operator, solution_update, rhs, heat_data.linear_solver, *trilinos_preconditioner);
      else
        AssertThrow(false, ExcNotImplemented());

      heat_operator->zero_out_ghost_values();
    };

    auto newton = NewtonRaphsonSolver<dim>(scratch_data,
                                           heat_data.nlsolve,
                                           temp_dof_idx,
                                           temp_quad_idx,
                                           temperature_old,
                                           temperature,
                                           create_rhs,
                                           solve_linear_system);

    newton.solve();

    distribute_constraints();

    ready_for_time_advance = false;
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::compute_interface_temperature(const VectorType &     distance,
                                                            const BlockVectorType &normal_vector)
  {
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(
      1e-6 /*tolerance*/, true /*unique mapping*/);

    LevelSet::Tools::broadcast_interface_value_to_vector<dim>(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx),
      scratch_data.get_dof_handler(temp_dof_idx),
      *level_set_as_heaviside,
      distance,
      normal_vector,
      temperature,
      temperature_interface,
      remote_point_evaluation);

    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(temperature_interface);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    temperature.update_ghost_values();
    vectors.push_back(&temperature);
    temperature_old.update_ghost_values();
    vectors.push_back(&temperature_old);
    temperature_interface.update_ghost_values();
    vectors.push_back(&temperature_interface);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::distribute_constraints()
  {
    scratch_data.get_constraint(temp_dof_idx).distribute(temperature);
    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(temperature_old);
    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(temperature_interface);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    /**
     *  temperature
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                             temperature,
                             "temperature");
    /**
     *  temperature old
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             temperature_old,
                             "temperature_old");
    /**
     *  heat source
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             heat_source,
                             "heat_source");
    /**
     *  evaporative heat source/sink
     */
    heat_operator->attach_output_vectors(data_out);
    /**
     *  temperature interface
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             temperature_interface,
                             "temperature_interface");
    /*
     *  user rhs
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             user_rhs,
                             "heat_user_rhs");
  }

  template <int dim>
  const VectorType &
  HeatTransferOperation<dim>::get_temperature() const
  {
    return temperature;
  }

  template <int dim>
  VectorType &
  HeatTransferOperation<dim>::get_temperature()
  {
    return temperature;
  }

  template <int dim>
  const VectorType &
  HeatTransferOperation<dim>::get_temperature_interface() const
  {
    return temperature_interface;
  }

  template <int dim>
  VectorType &
  HeatTransferOperation<dim>::get_temperature_interface()
  {
    return temperature_interface;
  }

  template <int dim>
  const VectorType &
  HeatTransferOperation<dim>::get_heat_source() const
  {
    return heat_source;
  }

  template <int dim>
  VectorType &
  HeatTransferOperation<dim>::get_heat_source()
  {
    return heat_source;
  }

  template <int dim>
  const VectorType &
  HeatTransferOperation<dim>::get_user_rhs() const
  {
    return user_rhs;
  }

  template <int dim>
  VectorType &
  HeatTransferOperation<dim>::get_user_rhs()
  {
    return user_rhs;
  }

  template <int dim>
  const VectorType &
  HeatTransferOperation<dim>::get_level_set_as_heaviside() const
  {
    Assert(
      level_set_as_heaviside,
      ExcMessage(
        "You requested the level set vector from the heat operation, which is a nullptr. You must provide a valid level_set_as_heaviside pointer to the heat operation."));
    return *level_set_as_heaviside;
  }

  template class HeatTransferOperation<1>;
  template class HeatTransferOperation<2>;
  template class HeatTransferOperation<3>;
} // namespace MeltPoolDG::Heat
