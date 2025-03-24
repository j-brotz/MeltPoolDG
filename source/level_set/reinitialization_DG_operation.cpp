#include <meltpooldg/level_set/reinitialization_DG_operation.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/iteration_monitor.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
namespace MeltPoolDG::LevelSet
{
  template <int dim>
  ReinitializationDGOperation<dim>::ReinitializationDGOperation(
    const ScratchData<dim>             &scratch_data_in,
    const ReinitializationData<double> &reinit_data,
    const TimeIterator<double>         &time_iterator,
    const unsigned int                  reinit_dof_idx_in,
    const unsigned int                  reinit_quad_idx_in,
    const unsigned int                  ls_dof_idx_in,
    const NormalVectorData<double>     &normal_vec_data,
    const CurvatureData<double>        &curvature_data)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data)
    , time_iterator(time_iterator)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , solution_history(std::max(reinit_data.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
  {
    normal_vector_operation =
      std::make_shared<NormalVectorDGOperation<dim>>(scratch_data_in,
                                                     reinit_dof_idx,
                                                     reinit_quad_idx,
                                                     solution_history.get_current_solution(),
                                                     normal_vec_data);

    curvature_operation = std::make_shared<CurvatureDGOperation<dim>>(
      scratch_data_in,
      reinit_quad_idx,
      reinit_quad_idx,
      normal_vector_operation->get_solution_normal_vector(),
      curvature_data);

    reinit_DG_operator = std::make_shared<ReinitilizationDGOperator<dim>>(
      scratch_data_in,
      reinit_data,
      reinit_dof_idx_in,
      reinit_quad_idx_in,
      curvature_operation->get_curvature(),
      normal_vector_operation->get_solution_normal_vector());

    reinitialization_integration =
      std::shared_ptr<TimeIntegratorBase<double>>(time_integrator_factory<double>(
        *reinit_DG_operator,
        reinit_data.reinitilization_DG_specific_data.time_integration_data,
        reinit_data.linear_solver,
        scratch_data_in.get_timer()));
  }

  template <int dim>
  ReinitializationDGOperation<dim>::ReinitializationDGOperation(
    const ScratchData<dim>                                &scratch_data_in,
    const ReinitializationData<double>                    &reinit_data,
    const TimeIterator<double>                            &time_iterator,
    const unsigned int                                     reinit_dof_idx_in,
    const unsigned int                                     reinit_quad_idx_in,
    const unsigned int                                     ls_dof_idx_in,
    const std::shared_ptr<NormalVectorOperationBase<dim>> &normal_vector_operation_in,
    const std::shared_ptr<CurvatureDGOperation<dim>>      &curvature_operation_in,
    const bool                                             is_coupled_in)
    : scratch_data(scratch_data_in)
    , reinit_data(reinit_data)
    , time_iterator(time_iterator)
    , reinit_dof_idx(reinit_dof_idx_in)
    , reinit_quad_idx(reinit_quad_idx_in)
    , ls_dof_idx(ls_dof_idx_in)
    , solution_history(std::max(reinit_data.predictor.n_old_solution_vectors,
                                2U /*TODO: include time integration scheme*/))
    , is_coupled(is_coupled_in)
  {
    normal_vector_operation = normal_vector_operation_in;

    curvature_operation = curvature_operation_in;

    reinit_DG_operator = std::make_shared<ReinitilizationDGOperator<dim>>(
      scratch_data_in,
      reinit_data,
      reinit_dof_idx_in,
      reinit_quad_idx_in,
      curvature_operation->get_curvature(),
      normal_vector_operation->get_solution_normal_vector());

    reinitialization_integration =
      std::shared_ptr<TimeIntegratorBase<double>>(time_integrator_factory<double>(
        *reinit_DG_operator,
        reinit_data.reinitilization_DG_specific_data.time_integration_data,
        reinit_data.linear_solver,
        scratch_data_in.get_timer()));
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::reinit()
  {
    solution_history.apply(
      [this](VectorType &v) { scratch_data.initialize_dof_vector(v, reinit_dof_idx); });

    reinit_DG_operator->reinit();
    reinitialization_integration->reinit(solution_history);

    if (!is_coupled)
      {
        normal_vector_operation->reinit();
        curvature_operation->reinit();
      }
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::set_initial_condition(
    const Function<dim> &initial_field_function)
  {
    FECellIntegrator<dim, 1, double> phi(scratch_data.get_matrix_free(),
                                         reinit_dof_idx,
                                         reinit_quad_idx);


    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, double> inverse(phi);

    if (solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().zero_out_ghost_values();

    for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            // Workaround for the Vectorized Array
            dealii::VectorizedArray<double> helper;
            for (unsigned int i = 0; i < VectorizedArray<double>::size(); i++)
              {
                if constexpr (dim == 1)
                  {
                    Point<dim> p(phi.quadrature_point(q)[0][i]);
                    helper[i] = initial_field_function.value(p);
                  }

                if (dim == 2)
                  {
                    Point<dim> p(phi.quadrature_point(q)[0][i], phi.quadrature_point(q)[1][i]);
                    helper[i] = initial_field_function.value(p);
                  }
                if (dim == 3)
                  {
                    Point<dim> p(phi.quadrature_point(q)[0][i],
                                 phi.quadrature_point(q)[1][i],
                                 phi.quadrature_point(q)[2][i]);
                    helper[i] = initial_field_function.value(p);
                  }
              }
            phi.submit_dof_value(helper, q);
          }
        inverse.transform_from_q_points_to_basis(1, phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(solution_history.get_current_solution());
      }

    if (solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().update_ghost_values();

    solution_history.commit_old_solutions();

    reinit_DG_operator->prepare_operator(solution_history.get_current_solution());
    reinit_DG_operator->set_artificial_diffusitivity();


    if (!is_coupled)
      {
        normal_vector_operation->solve();
        curvature_operation->solve();
      }
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::set_initial_condition(const VectorType &solution_level_set_in)
  {
    /*
     *    copy the given solution into the member variable
     */
    solution_history.get_current_solution().zero_out_ghost_values();
    solution_history.get_current_solution().copy_locally_owned_data_from(solution_level_set_in);


    if (solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().update_ghost_values();

    solution_history.commit_old_solutions();

    reinit_DG_operator->prepare_operator(solution_history.get_current_solution());
    reinit_DG_operator->set_artificial_diffusitivity();

    if (!is_coupled)
      {
        normal_vector_operation->solve();
        curvature_operation->solve();
      }
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::update_dof_idx(const unsigned int &reinit_dof_idx_in)
  {
    reinit_dof_idx = reinit_dof_idx_in;
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::init_time_advance()
  {
    if (solution_history.get_current_solution().has_ghost_elements())
      solution_history.get_current_solution().zero_out_ghost_values();
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::solve()
  {
    ScopedName         sc("reinitialization::solve");
    TimerOutput::Scope scope(scratch_data.get_timer(), sc);

    init_time_advance();

    if (reinit_data.linear_solver.do_matrix_free)
      {
        if (reinit_data.reinitilization_DG_specific_data.IMEX_integration_data.integrator_type !=
            TimeIntegratorSchemes::not_initialized)
          {
            reinit_DG_operator->apply_diffusion_implicit(time_iterator.get_old_time(),
                                                         time_iterator.get_current_time_increment(),
                                                         solution_history);
            solution_history.set_recent_old_solution(solution_history.get_current_solution());
          }

        reinitialization_integration->perform_time_step(time_iterator.get_old_time(),
                                                        time_iterator.get_current_time_increment(),
                                                        solution_history);
        solution_history.get_recent_old_solution().add(-1.0,
                                                       solution_history.get_current_solution());
        max_change_level_set = solution_history.get_recent_old_solution().linfty_norm();

        solution_history.commit_old_solutions();
      }
  }

  template <int dim>
  double
  ReinitializationDGOperation<dim>::get_max_change_level_set() const
  {
    return max_change_level_set;
  }

  template <int dim>
  const VectorType &
  ReinitializationDGOperation<dim>::get_level_set() const
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  VectorType &
  ReinitializationDGOperation<dim>::get_level_set()
  {
    return solution_history.get_current_solution();
  }

  template <int dim>
  const BlockVectorType &
  ReinitializationDGOperation<dim>::get_normal_vector() const
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim>
  BlockVectorType &
  ReinitializationDGOperation<dim>::get_normal_vector()
  {
    return normal_vector_operation->get_solution_normal_vector();
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    solution_history.apply([&](VectorType &v) { vectors.push_back(&v); });
    normal_vector_operation->attach_vectors(vectors);

    /**
     * When the mesh is refined the smoothed signum and the sign indicator function also need to be
     * refined.
     */
    vectors.push_back(&reinit_DG_operator->get_signum_smoothed());
    vectors.push_back(&reinit_DG_operator->get_sign_indicator_function());
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::attach_output_vectors(
    GenericDataOut<dim, double> &data_out) const
  {
    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                             solution_history.get_current_solution(),
                             "reinit_DG_psi");

    for (unsigned int d = 0; d < dim; ++d)
      data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                               get_normal_vector().block(d),
                               "normal_" + std::to_string(d));

    data_out.add_data_vector(scratch_data.get_dof_handler(reinit_dof_idx),
                             curvature_operation->get_curvature(),
                             "curvature");
  }

  template <int dim>
  double
  ReinitializationDGOperation<dim>::compute_CFL_based_timestep() const
  {
    const double minimal_vertex_distance = scratch_data.get_min_cell_size();
    const double fe_degree = ((static_cast<double>(scratch_data.get_degree(reinit_dof_idx))));

    if (reinit_data.reinitilization_DG_specific_data.IMEX_integration_data.integrator_type !=
        TimeIntegratorSchemes::not_initialized)
      {
        return reinit_data.reinitilization_DG_specific_data.CFL * minimal_vertex_distance /
               ((fe_degree * fe_degree) * reinit_DG_operator->get_signum_smoothed().linfty_norm());
      }
    else
      {
        return reinit_data.reinitilization_DG_specific_data.CFL /
               reinit_data.reinitilization_DG_specific_data.IP_diffusion /
               (std::pow(fe_degree, 2.0) / minimal_vertex_distance +
                reinit_DG_operator->get_max_diffusitivity() * std::pow(fe_degree, 4.0) /
                  (minimal_vertex_distance * minimal_vertex_distance));
      }
  }

  template <int dim>
  void
  ReinitializationDGOperation<dim>::set_artificial_diffusitivity()
  {
    reinit_DG_operator->set_artificial_diffusitivity();
  }

  template class ReinitializationDGOperation<1>;
  template class ReinitializationDGOperation<2>;
  template class ReinitializationDGOperation<3>;
} // namespace MeltPoolDG::LevelSet
