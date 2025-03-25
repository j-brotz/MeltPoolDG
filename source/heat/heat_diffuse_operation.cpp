#include <meltpooldg/heat/heat_diffuse_operation.hpp>
//

#include <deal.II/base/exceptions.h>

#include <deal.II/numerics/vector_tools_interpolate.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/utilities/constraints.hpp>
#include <meltpooldg/utilities/dof_monitor.hpp>
#include <meltpooldg/utilities/fe_util.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>


namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim, typename number>
  HeatDiffuseOperation<dim, number>::HeatDiffuseOperation(
    const ScratchData<dim, dim, number>                       &scratch_data_in,
    const std::shared_ptr<const BoundaryConditionManager<dim>> heat_bc_manager,
    const PeriodicBoundaryConditions<dim>                     &periodic_bc_in,
    const HeatData<number>                                    &heat_data_in,
    const Material<number>                                    &material,
    const TimeIterator<number>                                &time_iterator,
    const unsigned int                                         temp_dof_idx_in,
    const unsigned int                                         temp_hanging_nodes_dof_idx_in,
    const unsigned int                                         temp_quad_idx_in,
    const unsigned int                                         vel_dof_idx_in,
    const VectorType                                          *velocity_in,
    const unsigned int                                         ls_dof_idx_in,
    const VectorType                                          *level_set_as_heaviside_in,
    const bool                                                 do_solidification)
    : scratch_data(scratch_data_in)
    , dirichlet_bc(heat_bc_manager->get_bc_of_type("dirichlet"))
    , periodic_bc(periodic_bc_in)
    , heat_data(heat_data_in)
    , time_iterator(time_iterator)
    , temp_dof_idx(temp_dof_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , temp_quad_idx(temp_quad_idx_in)
    , vel_dof_idx(vel_dof_idx_in)
    , velocity(velocity_in)
    , ls_dof_idx(ls_dof_idx_in)
    , level_set_as_heaviside(level_set_as_heaviside_in)
    , newton(heat_data.nlsolve)
    , solution_history(std::max(heat_data.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
  {
    heat_operator = std::make_unique<HeatDiffuseMultiPhaseOperator<dim, number>>(
      scratch_data,
      heat_bc_manager,
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
      level_set_as_heaviside,
      do_solidification);

    preconditioner =
      make_preconditioner<dim, HeatDiffuseMultiPhaseOperator<dim, number>, VectorType>(
        heat_data.linear_solver.preconditioner_type, heat_operator.get());

    setup_newton();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::setup_newton()
  {
    newton.residual = [&](const VectorType & /*evaluation_point*/, VectorType &rhs) {
      // solely homogeneous dirichlet bc are distributed for the
      // corrected temperature field in the newton solver
      heat_operator->pre();
      rhs.copy_locally_owned_data_from(user_rhs);
      heat_operator->create_rhs(rhs, solution_history.get_recent_old_solution());
    };

    newton.solve_with_jacobian = [&](const VectorType &rhs, VectorType &solution_update) -> int {
      return LinearSolver::solve<VectorType, OperatorMatrixFree<dim, number>>(
        *heat_operator,
        solution_update,
        rhs,
        heat_data.linear_solver,
        preconditioner,
        "heat_operation");
    };

    newton.reinit_vector = [&](VectorType &v) {
      scratch_data.initialize_dof_vector(v, temp_dof_idx);
    };

    newton.distribute_constraints = [&](VectorType &v) {
      scratch_data.get_constraint(temp_dof_idx).distribute(v);
    };

    newton.norm_of_solution_vector = [this]() -> number {
      return MeltPoolDG::VectorTools::compute_norm<dim>(solution_history.get_current_solution(),
                                                        scratch_data,
                                                        temp_dof_idx,
                                                        temp_quad_idx);
    };
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::register_evaporative_mass_flux(
    VectorType                                 *evaporative_mass_flux_in,
    const unsigned int                          evapor_mass_flux_dof_idx_in,
    const Evaporation::EvaporationData<number> &evapor_data)
  {
    heat_operator->register_evaporative_mass_flux(evaporative_mass_flux_in,
                                                  evapor_mass_flux_dof_idx_in,
                                                  evapor_data);
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::register_surface_mesh(
    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<number> /*weights*/
                                 >> &surface_mesh_info_in)
  {
    heat_operator->register_surface_mesh(surface_mesh_info_in);
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::distribute_dofs(DoFHandler<dim> &dof_handler) const
  {
    FiniteElementUtils::distribute_dofs<dim, 1>(heat_data.fe, dof_handler);
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::setup_constraints(
    ScratchData<dim, dim, number> &mutable_scratch_data) const
  {
    Constraints::make_DBC_and_HNC_plus_PBC_and_merge_HNC_plus_PBC_into_DBC<dim>(
      mutable_scratch_data, dirichlet_bc, periodic_bc, temp_dof_idx, temp_hanging_nodes_dof_idx);
  }


  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::set_initial_condition(
    const Function<dim> &initial_field_function_temperature)
  {
    dealii::VectorTools::interpolate(scratch_data.get_mapping(),
                                     scratch_data.get_dof_handler(temp_dof_idx),
                                     initial_field_function_temperature,
                                     solution_history.get_current_solution());

    if (heat_data.enable_time_dependent_bc)
      MeltPoolDG::Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim>(
        const_cast<ScratchData<dim> &>(scratch_data),
        dirichlet_bc,
        temp_dof_idx,
        temp_hanging_nodes_dof_idx);

    scratch_data.get_constraint(temp_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.get_current_solution().update_ghost_values();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::reinit()
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

    heat_operator->reinit();

    preconditioner.reinit(scratch_data, temp_dof_idx);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::init_time_advance()
  {
    heat_operator->reset_time_increment(time_iterator.get_current_time_increment());

    if (heat_data.enable_time_dependent_bc)
      {
        MeltPoolDG::Constraints::make_DBC_and_HNC_and_merge_HNC_into_DBC<dim>(
          const_cast<ScratchData<dim> &>(scratch_data),
          dirichlet_bc,
          temp_dof_idx,
          temp_hanging_nodes_dof_idx);
      }

    if (!predictor)
      predictor = std::make_unique<Predictor<VectorType, number>>(heat_data.predictor,
                                                                  solution_history,
                                                                  &time_iterator);
    if (this->heat_data.predictor.type == PredictorType::least_squares_projection)
      {
        // solely homogeneous dirichlet bc are distributed for the
        // corrected temperature field in the newton solver
        heat_operator->pre();
        temp = user_rhs;
        heat_operator->create_rhs(
          temp, solution_history.get_current_solution() //= old_solution for current time step
        );
        heat_operator->post();
      }

    predictor->vmult(*heat_operator, temperature_extrapolated, temp);

    // apply constraints to predictor
    scratch_data.get_constraint(temp_dof_idx).distribute(solution_history.get_current_solution());

    ready_for_time_advance = true;
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::solve()
  {
    solve(true);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::solve(const bool do_finish_time_step)
  {
    if (!ready_for_time_advance)
      init_time_advance();

    if (!heat_data.linear_solver.do_matrix_free)
      AssertThrow(false, ExcNotImplemented());

    // setup preconditioner
    heat_operator->pre();
    preconditioner.update();

    try
      {
        newton.solve(solution_history.get_current_solution());
      }
    catch (const ExcNewtonDidNotConverge &)
      {
        AssertThrow(false, ExcHeatTransferNoConvergence());
      }

    if (do_finish_time_step)
      finish_time_advance();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::finish_time_advance()
  {
    heat_operator->post();
    solution_history.update_ghost_values();
    ready_for_time_advance = false;
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::compute_interface_temperature(
    const VectorType                         &distance,
    const BlockVectorType                    &normal_vector,
    const LevelSet::NearestPointData<number> &nearest_point_data)
  {
    if (!nearest_point_search)
      nearest_point_search = std::make_unique<LevelSet::Tools::NearestPoint<dim>>(
        scratch_data.get_mapping(),
        scratch_data.get_dof_handler(ls_dof_idx),
        distance,
        normal_vector,
        scratch_data.get_remote_point_evaluation(temp_hanging_nodes_dof_idx),
        nearest_point_data);

    nearest_point_search->reinit(scratch_data.get_dof_handler(temp_dof_idx));

    nearest_point_search->template fill_dof_vector_with_point_values(
      temperature_interface, scratch_data.get_dof_handler(temp_dof_idx), get_temperature());

    scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(temperature_interface);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::attach_vectors(std::vector<VectorType *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
    vectors.push_back(&temperature_interface);
    heat_operator->attach_vectors(vectors);
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::distribute_constraints()
  {
    scratch_data.get_constraint(temp_dof_idx).distribute(solution_history.get_current_solution());
    solution_history.apply_old([this](VectorType &v) {
      scratch_data.get_constraint(temp_hanging_nodes_dof_idx).distribute(v);
    });
    heat_operator->distribute_constraints();
  }

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::attach_output_vectors(GenericDataOut<dim> &data_out) const
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

  template <int dim, typename number>
  void
  HeatDiffuseOperation<dim, number>::attach_output_vectors_failed_step(
    GenericDataOut<dim> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             newton.get_solution_update(),
                             "temperature_newton_last_solution_update",
                             true /* force output */);
    data_out.add_data_vector(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                             newton.get_residual(),
                             "temperature_newton_failed_residual",
                             true /* force output */);
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_temperature() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_temperature()
  {
    return solution_history.get_current_solution();
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_temperature_interface() const
  {
    return temperature_interface;
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_temperature_interface()
  {
    return temperature_interface;
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_heat_source() const
  {
    return heat_source;
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_heat_source()
  {
    return heat_source;
  }

  template <int dim, typename number>
  const typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_user_rhs() const
  {
    return user_rhs;
  }

  template <int dim, typename number>
  typename HeatOperationBase<dim, number>::VectorType &
  HeatDiffuseOperation<dim, number>::get_user_rhs()
  {
    return user_rhs;
  }

  template class HeatDiffuseOperation<1, double>;
  template class HeatDiffuseOperation<2, double>;
  template class HeatDiffuseOperation<3, double>;
} // namespace MeltPoolDG::Heat
