#include <meltpooldg/heat/heat_transfer_operation.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/linear_solve.hpp>
#include <meltpooldg/utilities/newton_raphson_solver.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  HeatTransferOperation<dim>::HeatTransferOperation(
    const std::shared_ptr<BoundaryConditions<dim>> &bc_data,
    const ScratchData<dim> &                        scratch_data_in,
    const HeatData<double> &                        heat_data_in,
    const Material<double> &                        material,
    const unsigned int                              temp_dof_idx_in,
    const unsigned int                              temp_hanging_nodes_dof_idx_in,
    const unsigned int                              temp_quad_idx_in,
    const unsigned int                              vel_dof_idx_in,
    VectorType *                                    velocity_in,
    const unsigned int                              ls_dof_idx_in,
    VectorType *                                    level_set_as_heaviside_in)
    : scratch_data(scratch_data_in)
    , heat_data(heat_data_in)
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
    heat_transfer_preconditioner = std::make_shared<HeatTransferPreconditionerMatrixfree<dim>>(
      scratch_data, temp_dof_idx, heat_data.solver.preconditioner_type, heat_operator);

    reinit();
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::register_evaporative_mass_flux(
    VectorType *       evaporative_mass_flux_in,
    const unsigned int evapor_mass_flux_dof_idx_in,
    const double       latent_heat_of_evaporation)
  {
    heat_operator->register_evaporative_mass_flux(evaporative_mass_flux_in,
                                                  evapor_mass_flux_dof_idx_in,
                                                  latent_heat_of_evaporation);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::set_initial_condition(
    const Function<dim> &initial_field_function_temperature)
  {
    reinit();

    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(temp_dof_idx),
                                     initial_field_function_temperature,
                                     temperature);
    scratch_data.get_constraint(temp_dof_idx).distribute(temperature);
    temperature_old.copy_locally_owned_data_from(temperature);
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::reinit()
  {
    scratch_data.initialize_dof_vector(temperature, temp_dof_idx);
    scratch_data.initialize_dof_vector(temperature_old, temp_dof_idx);
    scratch_data.initialize_dof_vector(heat_source, temp_hanging_nodes_dof_idx);
    scratch_data.initialize_dof_vector(temperature_interface, temp_dof_idx);
    /*
     * setup sparsity pattern of system matrix only if the latter is
     * needed for computing the preconditioner
     */
    heat_transfer_preconditioner->reinit();
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::solve(const double dt)
  {
    if (!heat_data.do_matrix_free)
      AssertThrow(false, ExcNotImplemented());

    heat_operator->set_time_increment(dt);

    VectorType temperature_extrapolated;
    scratch_data.initialize_dof_vector(temperature_extrapolated, temp_dof_idx);

    UtilityFunctions::compute_linear_predictor(temperature,
                                               temperature_old,
                                               temperature_extrapolated,
                                               dt,
                                               dt); // @todo adapt for adaptive time stepping

    temperature_old = temperature;
    temperature     = temperature_extrapolated;

    const auto create_rhs = [&](VectorType &rhs) {
      // solely homogeneous dirichlet bc are distributed for the
      // corrected temperature field in the newton solver
      heat_operator->create_rhs(rhs, temperature_old);
    };

    const auto solve_linear_system = [&](VectorType &      solution_update,
                                         const VectorType &rhs) -> int {
      switch (heat_data.solver.preconditioner_type)
        {
          case PreconditionerType::Diagonal:
            case PreconditionerType::DiagonalReduced: {
              auto preconditioner = heat_transfer_preconditioner->compute_diagonal_preconditioner();

              return LinearSolve::solve<VectorType,
                                        SolverGMRES<VectorType>,
                                        OperatorBase<dim, double>>(*heat_operator,
                                                                   solution_update,
                                                                   rhs,
                                                                   heat_data.solver.rel_tolerance,
                                                                   heat_data.solver.max_iterations,
                                                                   preconditioner);
            }
          case PreconditionerType::Identity:
          case PreconditionerType::AMG:
          case PreconditionerType::AMGReduced:
          case PreconditionerType::ILU:
            case PreconditionerType::ILUReduced: {
              // take the first three letters as relevant preconditioner type
              auto preconditioner = heat_transfer_preconditioner->compute_trilinos_preconditioner();

              return LinearSolve::solve<VectorType,
                                        SolverGMRES<VectorType>,
                                        OperatorBase<dim, double>>(*heat_operator,
                                                                   solution_update,
                                                                   rhs,
                                                                   heat_data.solver.rel_tolerance,
                                                                   heat_data.solver.max_iterations,
                                                                   *preconditioner);
            }
            default: {
              AssertThrow(false, ExcNotImplemented());
              return 0;
            }
        }
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
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::distribute_constraints()
  {
    scratch_data.get_constraint(temp_dof_idx).distribute(temperature);
    scratch_data.get_constraint(temp_dof_idx).distribute(temperature_old);
    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(heat_source);
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
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
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
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                             temperature_interface,
                             "temperature_interface");
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
