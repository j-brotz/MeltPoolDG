#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>

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
    , solution_history(std::max(heat_data.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
  {
    heat_operator =
      std::make_shared<HeatTransferOperator<dim>>(bc_data,
                                                  scratch_data,
                                                  heat_data,
                                                  material,
                                                  temp_dof_idx,
                                                  temp_quad_idx,
                                                  temp_hanging_nodes_dof_idx,
                                                  solution_history.get_current_solution(),
                                                  solution_history.get_recent_old_solution(),
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
                                     solution_history.get_current_solution());

    if (heat_data.enable_time_dependent_bc)
      MeltPoolDG::UtilityFunctions::setup_and_merge_constraints<dim>(const_cast<ScratchData<dim> &>(
                                                                       scratch_data),
                                                                     bc_data->dirichlet_bc,
                                                                     temp_dof_idx,
                                                                     temp_hanging_nodes_dof_idx);

    scratch_data.get_constraint(temp_dof_idx).distribute(solution_history.get_current_solution());
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::reinit()
  {
    {
      ScopedName sc("heat::n_dofs");
      DoFMonitor::add_n_dofs(sc, scratch_data.get_dof_handler(temp_dof_idx).n_dofs());
    }
    scratch_data.initialize_dof_vector(solution_history.get_current_solution(), temp_dof_idx);
    solution_history.apply_old(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, temp_hanging_nodes_dof_idx); });

    scratch_data.initialize_dof_vector(user_rhs, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(heat_source, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(temperature_interface, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(temperature_extrapolated, temp_dof_idx);

    if (heat_data.predictor.type == PredictorType::least_squares_projection)
      scratch_data.initialize_dof_vector(temp, temp_hanging_nodes_dof_idx);

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

    if (!predictor)
      predictor = std::make_unique<Predictor<VectorType, double>>(heat_data.predictor,
                                                                  solution_history,
                                                                  &time_iterator);
    if (this->heat_data.predictor.type == PredictorType::least_squares_projection)
      {
        // solely homogeneous dirichlet bc are distributed for the
        // corrected temperature field in the newton solver
        heat_operator->update_ghost_values();
        temp = user_rhs;
        heat_operator->create_rhs(
          temp, solution_history.get_current_solution() //= old_solution for current time step
        );
        heat_operator->zero_out_ghost_values();
      }

    predictor->vmult(*heat_operator, temperature_extrapolated, temp);

    // apply constraints to predictor
    scratch_data.get_constraint(temp_dof_idx).distribute(solution_history.get_current_solution());
    ready_for_time_advance = true;
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::solve(const bool do_finish_time_step)
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

    // setup nonlinear solver
    auto newton = NewtonRaphsonSolver<dim>(heat_data.nlsolve);

    newton.residual = [&](const VectorType & /*evaluation_point*/, VectorType &rhs) {
      // solely homogeneous dirichlet bc are distributed for the
      // corrected temperature field in the newton solver
      heat_operator->update_ghost_values();
      rhs.copy_locally_owned_data_from(user_rhs);
      heat_operator->create_rhs(rhs, solution_history.get_recent_old_solution());
    };

    newton.solve_with_jacobian = [&](const VectorType &rhs, VectorType &solution_update) -> int {
      if (diag_preconditioner)
        return LinearSolver::solve<VectorType, OperatorBase<dim, double>>(
          *heat_operator, solution_update, rhs, heat_data.linear_solver, *diag_preconditioner);
      else if (trilinos_preconditioner)
        return LinearSolver::solve<VectorType, OperatorBase<dim, double>>(
          *heat_operator, solution_update, rhs, heat_data.linear_solver, *trilinos_preconditioner);
      else
        AssertThrow(false, ExcNotImplemented());
    };

    newton.reinit_vector = [&](VectorType &v) {
      scratch_data.initialize_dof_vector(v, temp_dof_idx);
    };

    newton.distribute_constraints = [&](VectorType &v) {
      scratch_data.get_constraint(temp_dof_idx).distribute(v);
    };

    newton.norm_of_solution_vector = [this]() -> double {
      return MeltPoolDG::VectorTools::compute_L2_norm<dim>(solution_history.get_current_solution(),
                                                           scratch_data,
                                                           temp_dof_idx,
                                                           temp_quad_idx);
    };

    try
      {
        newton.solve(solution_history.get_current_solution());
      }
    catch (const ExcNewtonDidNotConverge &e)
      {
        // TODO: move to problem
        DataOut<dim> data_out;

        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 solution_history.get_current_solution(),
                                 "solution");
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 newton.get_solution_update(),
                                 "solution_update");
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 newton.get_residual(),
                                 "rhs");
        data_out.build_patches(scratch_data.get_mapping());
        data_out.write_vtu_in_parallel("newton_raphson_failed.vtu", scratch_data.get_mpi_comm());

        AssertThrow(false, ExcNewtonDidNotConverge());
      }

    if (do_finish_time_step)
      finish_time_advance();
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::finish_time_advance()
  {
    ready_for_time_advance = false;
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::compute_interface_temperature(
    const VectorType &                        distance,
    const BlockVectorType &                   normal_vector,
    const ClosestPointProjectionData<double> &data)
  {
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(
      1e-6 /*tolerance*/, true /*unique mapping*/);

    LevelSet::Tools::broadcast_interface_value_to_vector<dim>(
      scratch_data.get_mapping(),
      scratch_data.get_dof_handler(ls_dof_idx),
      scratch_data.get_dof_handler(temp_dof_idx),
      distance,
      normal_vector,
      solution_history.get_current_solution(),
      temperature_interface,
      scratch_data.get_min_cell_size(ls_dof_idx),
      remote_point_evaluation,
      data.max_iter,
      data.rel_tol);

    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(temperature_interface);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
    vectors.push_back(&temperature_interface);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::distribute_constraints()
  {
    scratch_data.get_constraint(temp_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.apply_old([this](VectorType &v) {
      scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(v);
    });
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
  {
    /**
     *  temperature
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                             solution_history.get_current_solution(),
                             "temperature");
    /**
     *  temperature old
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             solution_history.get_recent_old_solution(),
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
    /*
     *  user rhs projected
     */
    if (data_out.is_requested("heat_user_rhs_projected"))
      {
        scratch_data.initialize_dof_vector(user_rhs_projected, temp_hanging_nodes_dof_idx);
        VectorTools::project_vector<1>(scratch_data.get_mapping(),
                                       scratch_data.get_dof_handler(temp_dof_idx),
                                       scratch_data.get_constraint(temp_dof_idx),
                                       scratch_data.get_quadrature(temp_quad_idx),
                                       user_rhs,
                                       user_rhs_projected);
        data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                                 user_rhs_projected,
                                 "heat_user_rhs_projected");
      }
  }

  template <int dim>
  const VectorType &
  HeatTransferOperation<dim>::get_temperature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  VectorType &
  HeatTransferOperation<dim>::get_temperature()
  {
    return solution_history.get_current_solution();
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
