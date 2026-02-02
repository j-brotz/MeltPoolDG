#include <meltpooldg/level_set/advection_DG_operator.hpp>
#include <meltpooldg/linear_algebra/utilities_matrixfree.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>
#include <meltpooldg/utilities/vector_tools.templates.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <int dim, typename number>
  AdvectionDGOperator<dim, number>::AdvectionDGOperator(
    const MeltPoolDG::ScratchData<dim, dim, number> &scratch_data_in,
    const unsigned int                               advec_diff_dof_idx_in,
    const unsigned int                               advec_diff_quad_idx_in,
    const std::shared_ptr<MeltPoolDG::BoundaryConditionManager<dim, number>>
      &boundary_conditions_in)
    : update_field_functions(true)
    , scratch_data_(scratch_data_in)
    , advec_diff_dof_idx(advec_diff_dof_idx_in)
    , advec_diff_quad_idx(advec_diff_quad_idx_in)
    , boundary_conditions(boundary_conditions_in)
  {}

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::update_time_velocity_function(const number time) const
  {
    advection_velocity_function->set_time(time);
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::set_advection_velocity(const VectorType  &advection_velocity_in,
                                                           const unsigned int velocity_dof_idx_in)
  {
    Assert(
      advection_velocity_function == nullptr,
      dealii::ExcMessage(
        "An advection velocity DoF vector was provided via set_advection_velocity(), "
        "but an advection velocity function has already been set via set_advection_velocity_function(). "
        "Please use only one of these methods to specify the advection velocity."));

    advection_velocity_ = &advection_velocity_in;
    velocity_dof_idx    = velocity_dof_idx_in;
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::set_advection_velocity_function(
    const std::shared_ptr<dealii::Function<dim, number>> &advection_velocity_function_in)
  {
    Assert(
      advection_velocity_ == nullptr,
      dealii::ExcMessage(
        "An advection velocity function was provided via set_advection_velocity_function(), "
        "but an advection velocity DoF vector has already been set via set_advection_velocity(). "
        "Please use only one of these methods to specify the advection velocity."));

    advection_velocity_function = advection_velocity_function_in;
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

    std::unique_ptr<FECellIntegrator<dim, dim, number>> eval_vel;

    if (advection_velocity_)
      eval_vel = std::make_unique<FECellIntegrator<dim, dim, number>>(data,
                                                                      velocity_dof_idx,
                                                                      advec_diff_quad_idx);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.gather_evaluate(src, EvaluationFlags::values);

        if (advection_velocity_)
          {
            eval_vel->reinit(cell);
            eval_vel->gather_evaluate(*advection_velocity_, EvaluationFlags::values);
          }

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const vector speed = (advection_velocity_) ?
                                   MeltPoolDG::VectorTools::to_vector<dim>(eval_vel->get_value(q)) :
                                   VectorTools::evaluate_function_at_vectorized_points<dim, number>(
                                     *advection_velocity_function, eval.quadrature_point(q));
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

    std::unique_ptr<FEFaceIntegrator<dim, dim, number>> eval_vel;

    if (advection_velocity_)
      eval_vel = std::make_unique<FEFaceIntegrator<dim, dim, number>>(data,
                                                                      true,
                                                                      velocity_dof_idx,
                                                                      advec_diff_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_plus.gather_evaluate(src, EvaluationFlags::values);

        if (advection_velocity_)
          {
            eval_vel->reinit(face);
            eval_vel->gather_evaluate(*advection_velocity_, EvaluationFlags::values);
          }

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const vector speed =
              (advection_velocity_) ?
                MeltPoolDG::VectorTools::to_vector<dim>(eval_vel->get_value(q)) :
                VectorTools::evaluate_function_at_vectorized_points<dim, number, dim>(
                  *advection_velocity_function, eval_minus.quadrature_point(q));
            const auto   u_minus = eval_minus.get_value(q);
            const auto   u_plus  = eval_plus.get_value(q);
            const vector normal_vector_minus =
              MeltPoolDG::VectorTools::to_vector<dim>(eval_minus.normal_vector(q));

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

    std::unique_ptr<FEFaceIntegrator<dim, dim, number>> eval_vel;

    if (advection_velocity_)
      eval_vel = std::make_unique<FEFaceIntegrator<dim, dim, number>>(data,
                                                                      true,
                                                                      velocity_dof_idx,
                                                                      advec_diff_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        if (advection_velocity_)
          {
            eval_vel->reinit(face);
            eval_vel->gather_evaluate(*advection_velocity_, EvaluationFlags::values);
          }

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const vector speed =
              (advection_velocity_) ?
                MeltPoolDG::VectorTools::to_vector<dim>(eval_vel->get_value(q)) :
                VectorTools::evaluate_function_at_vectorized_points<dim, number, dim>(
                  *advection_velocity_function, eval_minus.quadrature_point(q));

            const auto   u_minus = eval_minus.get_value(q);
            const vector normal_vector =
              MeltPoolDG::VectorTools::to_vector<dim>(eval_minus.normal_vector(q));

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

    std::unique_ptr<FEFaceIntegrator<dim, dim, number>> eval_vel;

    if (advection_velocity_)
      eval_vel = std::make_unique<FEFaceIntegrator<dim, dim, number>>(data,
                                                                      true,
                                                                      velocity_dof_idx,
                                                                      advec_diff_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        if (advection_velocity_)
          {
            eval_vel->reinit(face);
            eval_vel->gather_evaluate(*advection_velocity_, EvaluationFlags::values);
          }

        const auto boundary_id = data.get_boundary_id(face);

        if (boundary_conditions->get_type(boundary_id) == "dirichlet")
          {
            for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
              {
                const vector speed =
                  (advection_velocity_) ?
                    MeltPoolDG::VectorTools::to_vector<dim>(eval_vel->get_value(q)) :
                    VectorTools::evaluate_function_at_vectorized_points<dim, number, dim>(
                      *advection_velocity_function, eval_minus.quadrature_point(q));

                const auto   u_minus = eval_minus.get_value(q);
                const vector normal_vector =
                  MeltPoolDG::VectorTools::to_vector<dim>(eval_minus.normal_vector(q));

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
    if (boundary_conditions)
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

    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type inverse =
      [dof_idx  = advec_diff_dof_idx,
       quad_idx = advec_diff_quad_idx](const MatrixFree<dim, number>              &matrix_free,
                                       LinearAlgebra::distributed::Vector<number> &dst,
                                       const LinearAlgebra::distributed::Vector<number> &src,
                                       const std::pair<unsigned int, unsigned int> cell_range) {
        Utilities::MatrixFree::local_apply_inverse_mass_matrix<dim, 1, number>(
          matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };

    this->scratch_data_.get_matrix_free().cell_loop(
      inverse, dst, dst, std::function<void(unsigned int, unsigned int)>(), func);
  }

  template <int dim, typename number>
  void
  AdvectionDGOperator<dim, number>::apply_dirichlet_boundary_operator(
    const number                                      time,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    if (boundary_conditions)
      {
        boundary_conditions->set_time(time);
      }

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

    using local_applier_type =
      std::function<void(const dealii::MatrixFree<dim, number> &,
                         dealii::LinearAlgebra::distributed::Vector<number>       &dst,
                         const dealii::LinearAlgebra::distributed::Vector<number> &src,
                         const std::pair<unsigned int, unsigned int> &)>;

    local_applier_type inverse =
      [dof_idx  = advec_diff_dof_idx,
       quad_idx = advec_diff_quad_idx](const MatrixFree<dim, number>              &matrix_free,
                                       LinearAlgebra::distributed::Vector<number> &dst,
                                       const LinearAlgebra::distributed::Vector<number> &src,
                                       const std::pair<unsigned int, unsigned int> cell_range) {
        Utilities::MatrixFree::local_apply_inverse_mass_matrix<dim, 1, number>(
          matrix_free, dst, src, cell_range, dof_idx, quad_idx);
      };

    this->scratch_data_.get_matrix_free().cell_loop(inverse, dst, dst);
  }

  template class AdvectionDGOperator<1, double>;
  template class AdvectionDGOperator<2, double>;
  template class AdvectionDGOperator<3, double>;
} // namespace MeltPoolDG::LevelSet
