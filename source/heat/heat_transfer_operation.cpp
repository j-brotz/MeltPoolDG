#include <meltpooldg/heat/heat_transfer_operation.hpp>
//

#include <meltpooldg/utilities/linear_solve.hpp>
#include <meltpooldg/utilities/newton_raphson_solver.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  HeatTransferOperation<dim>::HeatTransferOperation(
    const std::shared_ptr<BoundaryConditions<dim>> &bc_data,
    const ScratchData<dim> &                        scratch_data_in,
    const HeatData<double> &                        heat_data_in,
    const MaterialData<double> &                    material_data,
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
    , material_data(material_data)
    , heat_transfer_preconditioner(scratch_data, temp_dof_idx)
  {
    heat_operator = std::make_shared<HeatTransferOperator<dim>>(bc_data,
                                                                scratch_data,
                                                                heat_data,
                                                                material_data,
                                                                temp_dof_idx,
                                                                temp_quad_idx,
                                                                temperature,
                                                                temperature_old,
                                                                heat_source,
                                                                vel_dof_idx,
                                                                velocity,
                                                                ls_dof_idx,
                                                                level_set_as_heaviside);

    reinit();
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::register_evaporative_mass_flux(
    VectorType * evaporative_mass_flux_in,
    const double latent_heat_of_evaporation)
  {
    heat_operator->register_evaporative_mass_flux(evaporative_mass_flux_in,
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
    scratch_data.initialize_dof_vector(heat_source, temp_dof_idx);
    /*
     * setup sparsity pattern of system matrix only if the latter is
     * needed for computing the preconditioner
     */
    if (heat_data.solver.preconditioner_type == "Diagonal" ||
        heat_data.solver.preconditioner_type == "AMG" ||
        heat_data.solver.preconditioner_type == "AMGReduced" ||
        heat_data.solver.preconditioner_type == "ILU" ||
        heat_data.solver.preconditioner_type == "ILUReduced")
      heat_transfer_preconditioner.reinit();
  }

  template <int dim>
  void
  HeatTransferOperation<dim>::solve(const double dt)
  {
    if (!heat_data.do_matrix_free)
      AssertThrow(false, ExcNotImplemented());

    heat_operator->set_time_increment(dt);
    temperature_old = temperature;

    const auto create_rhs = [&](VectorType &rhs) {
      // solely homogeneous dirichlet bc are distributed for the
      // corrected temperature field in the newton solver
      heat_operator->create_rhs(rhs, temperature_old);
    };

    const auto solve_linear_system = [&](VectorType &      solution_update,
                                         const VectorType &rhs) -> int {
      if (heat_data.solver.preconditioner_type == "Identity")
        {
          return LinearSolve::solve<VectorType, SolverGMRES<VectorType>, OperatorBase<dim, double>>(
            *heat_operator,
            solution_update,
            rhs,
            heat_data.solver.rel_tolerance,
            heat_data.solver.max_iterations);
        }
      else if (heat_data.solver.preconditioner_type == "Diagonal" ||
               heat_data.solver.preconditioner_type == "DiagonalReduced")

        {
          auto preconditioner = heat_transfer_preconditioner.get_diagonal_preconditioner(
            heat_data.solver.preconditioner_type, heat_operator);

          return LinearSolve::solve<VectorType, SolverGMRES<VectorType>, OperatorBase<dim, double>>(
            *heat_operator,
            solution_update,
            rhs,
            heat_data.solver.rel_tolerance,
            heat_data.solver.max_iterations,
            preconditioner);
        }
      else if (heat_data.solver.preconditioner_type == "AMG" ||
               heat_data.solver.preconditioner_type == "AMGReduced" ||
               heat_data.solver.preconditioner_type == "ILU" ||
               heat_data.solver.preconditioner_type == "ILUReduced")
        {
          const std::string precondition_base_type =
            heat_data.solver.preconditioner_type.find("AMG") != std::string::npos ? "AMG" : "ILU";
          heat_operator->compute_system_matrix(heat_transfer_preconditioner.get_system_matrix(),
                                               heat_data.solver.preconditioner_type ==
                                                 precondition_base_type);

          // take the first three letters as relevant preconditioner type
          auto preconditioner =
            LinearSolve::setup_preconditioner(heat_transfer_preconditioner.get_system_matrix(),
                                              precondition_base_type);

          return LinearSolve::solve<VectorType, SolverGMRES<VectorType>, OperatorBase<dim, double>>(
            *heat_operator,
            solution_update,
            rhs,
            heat_data.solver.rel_tolerance,
            heat_data.solver.max_iterations,
            *preconditioner);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
          return 0;
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
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                             heat_source,
                             "heat_source");
    /**
     *  evaporative heat source/sink
     */
    heat_operator->attach_output_vectors(data_out);
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
