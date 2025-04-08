#include <meltpooldg/level_set/advection_DG_operator.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  AdvectionDGOperator<dim, number>::AdvectionDGOperator(
    const MeltPoolDG::ScratchData<dim, dim, number>                         &scratch_data_in,
    VectorType                                                              &advection_velocity_in,
    const unsigned int                                                       advec_diff_dof_idx_in,
    const unsigned int                                                       advec_diff_quad_idx_in,
    const unsigned int                                                       velocity_dof_idx_in,
    const std::shared_ptr<MeltPoolDG::BoundaryConditionManager<dim, number>> boundary_conditions_in,
    std::shared_ptr<dealii::Function<dim>>                                  &advection_field_in,
    bool const enable_analytical_velocity_update_in)
    : update_field_functions(true)
    , scratch_data_(scratch_data_in)
    , advection_velocity_(advection_velocity_in)
    , advec_diff_dof_idx(advec_diff_dof_idx_in)
    , advec_diff_quad_idx(advec_diff_quad_idx_in)
    , velocity_dof_idx(velocity_dof_idx_in)
    , boundary_conditions(boundary_conditions_in)
    , advection_field(advection_field_in)
    , enable_analytical_velocity_update(enable_analytical_velocity_update_in)
  {}

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::set_field_functions(const number time) const
  {
    if (enable_analytical_velocity_update)
      {
        scratch_data_.initialize_dof_vector(advection_velocity_, velocity_dof_idx);
        // et the current time to the advection field function
        advection_field->set_time(time);
        // interpolate the values of the advection velocity
        dealii::VectorTools::interpolate(scratch_data_.get_mapping(),
                                         scratch_data_.get_dof_handler(velocity_dof_idx),
                                         *advection_field,
                                         advection_velocity_);
      }
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::local_apply_domain(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FECellIntegrator<dim, 1, number> eval(data, advec_diff_dof_idx, advec_diff_quad_idx);

    FECellIntegrator<dim, dim, number> eval_vel(data, velocity_dof_idx, advec_diff_quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval_vel.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::values);
        eval_vel.gather_evaluate(advection_velocity_, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const vector speed = MeltPoolDG::VectorTools::to_vector<dim>(eval_vel.get_value(q));
            const scalar u     = eval.get_value(q);
            const vector flux  = speed * u;
            eval.submit_gradient(flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::local_apply_inner_face(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceIntegrator<dim, 1, number> eval_minus(data,
                                                true,
                                                advec_diff_dof_idx,
                                                advec_diff_quad_idx);

    FEFaceIntegrator<dim, 1, number> eval_plus(data,
                                               false,
                                               advec_diff_dof_idx,
                                               advec_diff_quad_idx);

    FEFaceIntegrator<dim, dim, number> eval_vel(data, true, velocity_dof_idx, advec_diff_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_vel.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_plus.gather_evaluate(src, EvaluationFlags::values);
        eval_vel.gather_evaluate(advection_velocity_, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const vector speed   = MeltPoolDG::VectorTools::to_vector<dim>(eval_vel.get_value(q));
            const auto   u_minus = eval_minus.get_value(q);
            const auto   u_plus  = eval_plus.get_value(q);
            const vector normal_vector_minus =
              MeltPoolDG::VectorTools::to_vector<dim>(eval_minus.get_normal_vector(q));

            const scalar normal_times_speed = speed * normal_vector_minus;

            const auto flux_times_normal_of_minus =
              0.5 * ((u_minus + u_plus) * normal_times_speed +
                     std::abs(normal_times_speed) * (u_minus - u_plus));

            eval_minus.submit_value(-flux_times_normal_of_minus, q);
            eval_plus.submit_value(flux_times_normal_of_minus, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::local_apply_homogenous_boundary_face(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceIntegrator<dim, 1, number> eval_minus(data,
                                                true,
                                                advec_diff_dof_idx,
                                                advec_diff_quad_idx);

    FEFaceIntegrator<dim, dim, number> eval_vel(data, true, velocity_dof_idx, advec_diff_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_vel.reinit(face);
        eval_vel.gather_evaluate(advection_velocity_, EvaluationFlags::values);


        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const vector speed = MeltPoolDG::VectorTools::to_vector<dim>(eval_vel.get_value(q));

            const auto   u_minus = eval_minus.get_value(q);
            const vector normal_vector =
              MeltPoolDG::VectorTools::to_vector<dim>(eval_minus.get_normal_vector(q));

            dealii::VectorizedArray<number> u_plus;
            const auto                      boundary_id = data.get_boundary_id(face);
            if (boundary_conditions->get_type(boundary_id) == "dirichlet")
              {
                u_plus = u_minus * 0.0;
              }
            else
              {
                u_plus = u_minus;
              }

            // Compute the flux
            const scalar normal_times_speed = speed * normal_vector;

            // Homogenous boundary with  mirror principle
            const auto flux_times_normal =
              0.5 * ((u_minus + u_plus) * normal_times_speed +
                     std::abs(normal_times_speed) * (u_minus - u_plus));

            eval_minus.submit_value(-flux_times_normal, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::local_apply_inhomogenous_boundary_face(
    [[maybe_unused]] const MatrixFree<dim, number>                    &data,
    [[maybe_unused]] LinearAlgebra::distributed::Vector<number>       &dst,
    [[maybe_unused]] const LinearAlgebra::distributed::Vector<number> &src,
    [[maybe_unused]] const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceIntegrator<dim, 1, number> eval_minus(data,
                                                true,
                                                advec_diff_dof_idx,
                                                advec_diff_quad_idx);

    FEFaceIntegrator<dim, dim, number> eval_vel(data, true, velocity_dof_idx, advec_diff_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_vel.reinit(face);
        eval_vel.gather_evaluate(advection_velocity_, EvaluationFlags::values);

        const auto boundary_id = data.get_boundary_id(face);
        if (boundary_conditions->get_type(boundary_id) == "dirichlet")
          {
            for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
              {
                const vector speed = MeltPoolDG::VectorTools::to_vector<dim>(eval_vel.get_value(q));

                const auto   u_minus = eval_minus.get_value(q);
                const vector normal_vector =
                  MeltPoolDG::VectorTools::to_vector<dim>(eval_minus.get_normal_vector(q));

                // dealii::VectorizedArray<number> u_plus;
                auto const u_plus =
                  VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                    *(boundary_conditions->get_bc_of_type("dirichlet", false /*not optional*/)
                        .at(boundary_id)),
                    eval_minus.quadrature_point(q),
                    0);

                // Compute the flux
                const scalar normal_times_speed = speed * normal_vector;
                const auto   flux_times_normal =
                  0.5 * ((u_minus + u_plus) * normal_times_speed +
                         std::abs(normal_times_speed) * (u_minus - u_plus));

                eval_minus.submit_value(-flux_times_normal, q);
              }
            eval_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
      }
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::apply_operator(
    const number                                           time,
    LinearAlgebra::distributed::Vector<number>            &dst,
    LinearAlgebra::distributed::Vector<number> const      &src,
    const std::function<void(unsigned int, unsigned int)> &func) const
  {
    boundary_conditions->set_time(time);

    this->scratch_data_.get_matrix_free().loop(
      &AdvectionDGOperator<dim, number>::local_apply_domain,
      &AdvectionDGOperator<dim, number>::local_apply_inner_face,
      &AdvectionDGOperator<dim, number>::local_apply_homogenous_boundary_face,
      this,
      dst,
      src,
      false,
      MatrixFree<dim, number>::DataAccessOnFaces::values,
      MatrixFree<dim, number>::DataAccessOnFaces::values);

    this->scratch_data_.get_matrix_free().cell_loop(
      &AdvectionDGOperator<dim, number>::local_apply_inverse_mass_matrix,
      this,
      dst,
      dst,
      std::function<void(unsigned int, unsigned int)>(),
      func);
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::apply_dirichlet_boundary_operator(
    const number                                      time,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    boundary_conditions->set_time(time);

    scratch_data_.get_matrix_free().loop(
      &AdvectionDGOperator<dim, number>::local_apply_domain_dummy,
      &AdvectionDGOperator<dim, number>::local_apply_inner_face_dummy,
      &AdvectionDGOperator<dim, number>::local_apply_inhomogenous_boundary_face,
      this,
      dst,
      src,
      false,
      MatrixFree<dim, number>::DataAccessOnFaces::values,
      MatrixFree<dim, number>::DataAccessOnFaces::values);


    this->scratch_data_.get_matrix_free().cell_loop(
      &AdvectionDGOperator<dim, number>::local_apply_inverse_mass_matrix, this, dst, dst);
  }


  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FECellIntegrator<dim, 1, number> eval(data, advec_diff_dof_idx, advec_diff_quad_idx);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, number> inverse(eval);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);

        inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

        eval.set_dof_values(dst);
      }
  }

  template class AdvectionDGOperator<1, double>;
  template class AdvectionDGOperator<2, double>;
  template class AdvectionDGOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
