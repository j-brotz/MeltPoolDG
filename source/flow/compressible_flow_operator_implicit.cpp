#include <meltpooldg/flow/compressible_flow_operator_implicit.hpp>
#include <meltpooldg/time_integration/time_integrator_util.hpp>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  CompressibleFlowOperatorImplicit<dim, number>::CompressibleFlowOperatorImplicit(
    const CompressibleFlowData                     &compressible_flow_data_in,
    const ScratchData<dim>                         &scratch_data_in,
    ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
    const unsigned int                              comp_flow_dof_idx_in,
    const unsigned int                              comp_flow_quad_idx_in)
    : CompressibleFlowOperatorImplicitBase<dim, number>(compressible_flow_data_in,
                                                        scratch_data_in,
                                                        solution_history_in,
                                                        comp_flow_dof_idx_in,
                                                        comp_flow_quad_idx_in)
  {
    time_integrator =
      std::unique_ptr<TimeIntegratorBase<number, CompressibleFlowOperatorImplicit<dim, number>>>(
        implicit_time_integrator_factory<number, CompressibleFlowOperatorImplicit<dim, number>>(
          this->comp_flow_data.time_integrator, this->scratch_data.get_timer()));
    this->solution_history.resize(time_integrator->required_solution_history_size());
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::advance_time_step(
    number                                                        current_time,
    number                                                        time_step,
    std::function<void(number, VectorType &, const VectorType &)> pre_processing,
    std::function<void(number, VectorType &, const VectorType &)> post_processing)
  {
    time_integrator->perform_time_step(
      *this, current_time, time_step, this->solution_history, pre_processing, post_processing);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::reinit()
  {
    if (this->comp_flow_data.jacobian_type == "finite_difference")
      this->scratch_data.initialize_dof_vector(disturbed_residual, this->comp_flow_dof_idx);
    this->calculate_interior_penalty_parameter();
    time_integrator->reinit(this->solution_history);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::set_stage_constants(
    const number      current_time,
    const number      time_step,
    const VectorType &old_solution_in,
    const number      rhs_scaling_factor) const
  {
    this->update_boundary_conditions(current_time);

    current_time_step            = time_step;
    residual_rhs_scaling_factor  = rhs_scaling_factor;
    time_integrator_old_solution = &old_solution_in;
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::compute_residual(number,
                                                                  const VectorType &src,
                                                                  VectorType       &dst) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());
    this->scratch_data.get_matrix_free().loop(
      &CompressibleFlowOperatorImplicit::local_cell_residual,
      &CompressibleFlowOperatorImplicit::local_face_residual,
      &CompressibleFlowOperatorImplicit::local_boundary_face_residual,
      this,
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::apply_jacobian(VectorType       &dst,
                                                                const VectorType &src) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());
    if (this->comp_flow_data.jacobian_type == "exact")
      apply_jacobian_analytic(src, dst);
    else if (this->comp_flow_data.jacobian_type == "finite_difference")
      apply_jacobian_finite_differences(src, dst);
    else
      AssertThrow(false,
                  dealii::ExcMessage("The provided jacobian type '" +
                                     this->comp_flow_data.jacobian_type + "' is not supported!"));
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::apply_jacobian_analytic(const VectorType &src,
                                                                         VectorType &dst) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());
    this->scratch_data.get_matrix_free().loop(
      &CompressibleFlowOperatorImplicit::local_cell_jacobian,
      &CompressibleFlowOperatorImplicit::local_face_jacobian,
      &CompressibleFlowOperatorImplicit::local_boundary_face_jacobian,
      this,
      dst,
      src,
      true);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::apply_jacobian_finite_differences(
    const VectorType &src,
    VectorType       &dst) const
  {
    Assert(dst.partitioners_are_globally_compatible(*(src.get_partitioner())),
           typename VectorType::ExcVectorTypeNotCompatible());
    constexpr number sqrt_machine_precision = std::sqrt(1.0e-16);
    const number     src_l2_norm            = src.l2_norm();
    const number     epsilon = sqrt_machine_precision / (src_l2_norm > 0 ? src_l2_norm : 1.);

    auto local_compute_residual = [&](const VectorType &src_vec,
                                      VectorType       &dst_vec,
                                      const bool        zero_dst_vec,
                                      const number      factor) -> void {
      this->scratch_data.get_matrix_free().loop(
        &CompressibleFlowOperatorImplicit::local_cell_residual,
        &CompressibleFlowOperatorImplicit::local_face_residual,
        &CompressibleFlowOperatorImplicit::local_boundary_face_residual,
        this,
        dst_vec,
        src_vec,
        [&dst_vec, zero_dst_vec](const unsigned int begin_range,
                                 const unsigned int end_range) -> void {
          if (zero_dst_vec)
            for (unsigned int i = begin_range; i < end_range; ++i)
              dst_vec.local_element(i) = 0.0;
        },
        [&dst_vec, factor](const unsigned int begin_range, const unsigned end_range) -> void {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (unsigned int i = begin_range; i < end_range; ++i)
            dst_vec.local_element(i) *= factor;
        });
    };

    local_compute_residual(this->solution_history.get_current_solution(), dst, true, 1.0);
    DEAL_II_OPENMP_SIMD_PRAGMA
    for (unsigned int i = 0; i < disturbed_residual.locally_owned_size(); ++i)
      disturbed_residual.local_element(i) =
        this->solution_history.get_current_solution().local_element(i) +
        epsilon * src.local_element(i);
    local_compute_residual(disturbed_residual, dst, false, -1.0 / epsilon);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::local_cell_residual(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(this->scratch_data.get_matrix_free(),
                                               this->comp_flow_dof_idx,
                                               this->comp_flow_quad_idx);
    FECellIntegrator<dim, dim + 2, number> phi_old(this->scratch_data.get_matrix_free(),
                                                   this->comp_flow_dof_idx,
                                                   this->comp_flow_quad_idx);

    Tensor<1, dim, VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim> *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(this->body_force.get());

    if (constant_function)
      constant_body_force =
        VectorTools::evaluate_function_at_vectorized_points(*constant_function,
                                                            Point<dim, VectorizedArray<number>>());


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_old.reinit(cell);
        phi_old.gather_evaluate(*time_integrator_old_solution, dealii::EvaluationFlags::values);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            auto [value_q, grad_q] =
              this->template rhs_cell_integral_kernel<FECellIntegrator<dim, dim + 2, number>>(
                phi, q, constant_function ? &constant_body_force : nullptr);
            grad_q *= residual_rhs_scaling_factor;

            value_q *= residual_rhs_scaling_factor;
            value_q -= 1 / current_time_step * (phi.get_value(q) - phi_old.get_value(q));

            phi.submit_gradient(grad_q, q);
            phi.submit_value(value_q, q);
          }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::local_face_residual(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(this->scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 this->comp_flow_dof_idx,
                                                 this->comp_flow_quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(this->scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 this->comp_flow_dof_idx,
                                                 this->comp_flow_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        const VectorizedArray<number> penalty_parameter =
          (this->comp_flow_data.dynamic_viscosity > 0) ?
            std::max(phi_m.read_cell_data(this->interior_penalty_parameter),
                     phi_p.read_cell_data(this->interior_penalty_parameter)) :
            0.;


        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            auto [flux_m, flux_p, grad_flux_m, grad_flux_p] =
              this->template rhs_face_integral_kernel<FEFaceIntegrator<dim, dim + 2, number>>(
                phi_m, phi_p, q, penalty_parameter);


            // since we approach the face only once, we submit the contributions
            // to the face integral of the two neighbouring elements.
            phi_m.submit_gradient(residual_rhs_scaling_factor * grad_flux_m, q);
            phi_p.submit_gradient(residual_rhs_scaling_factor * grad_flux_p, q);
            phi_m.submit_value(residual_rhs_scaling_factor * flux_m, q);
            phi_p.submit_value(residual_rhs_scaling_factor * flux_p, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values |
                                  ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values |
                                  ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                     EvaluationFlags::gradients :
                                     EvaluationFlags::nothing),
                                dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::local_boundary_face_residual(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi(this->scratch_data.get_matrix_free(),
                                               true,
                                               this->comp_flow_dof_idx,
                                               this->comp_flow_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const VectorizedArray<number> penalty_parameter =
          (this->comp_flow_data.dynamic_viscosity > 0) ?
            phi.read_cell_data(this->interior_penalty_parameter) :
            0.;

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto [flux_m, grad_flux_m] = this->template rhs_boundary_face_integral_kernel<
              FEFaceIntegrator<dim, dim + 2, number>>(phi, q, face, penalty_parameter);

            phi.submit_value(residual_rhs_scaling_factor * flux_m, q);
            phi.submit_gradient(residual_rhs_scaling_factor * grad_flux_m, q);
          }
        phi.integrate_scatter(EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing),
                              dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::local_cell_jacobian(
    const MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(this->scratch_data.get_matrix_free(),
                                               this->comp_flow_dof_idx,
                                               this->comp_flow_quad_idx);
    FECellIntegrator<dim, dim + 2, number> delta_phi(this->scratch_data.get_matrix_free(),
                                                     this->comp_flow_dof_idx,
                                                     this->comp_flow_quad_idx);

    Tensor<1, dim, VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim> *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(this->body_force.get());

    if (constant_function)
      constant_body_force =
        VectorTools::evaluate_function_at_vectorized_points(*constant_function,
                                                            Point<dim, VectorizedArray<number>>());

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(this->solution_history.get_current_solution(),
                            EvaluationFlags::values | EvaluationFlags::gradients);
        delta_phi.reinit(cell);
        delta_phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto w_q       = phi.get_value(q);
            const auto delta_w_q = delta_phi.get_value(q);

            // time derivative
            ConservedVariablesType differential_change_time_derivative =
              1. / current_time_step * delta_w_q;
            delta_phi.submit_value(differential_change_time_derivative, q);

            dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> forcing;
            if (this->body_force.get() != nullptr)
              {
                const Tensor<1, dim, VectorizedArray<number>> force =
                  constant_function ?
                    constant_body_force :
                    VectorTools::evaluate_function_at_vectorized_points(*this->body_force,
                                                                        phi.quadrature_point(q));
                for (unsigned int d = 0; d < dim; ++d)
                  forcing[dim + 1] +=
                    force[d] * (delta_w_q[d + 1] * 1. / w_q[0] - w_q[d + 1] * delta_w_q[0]);
              }

            // convective flux
            ConservedVariablesGradType differential_change_flux =
              -1.0 * this->calculate_jacobain_convective_flux(w_q, delta_w_q);

            // viscous flux
            if (this->comp_flow_data.dynamic_viscosity > 0)
              {
                const auto grad_w_q       = phi.get_gradient(q);
                const auto grad_delta_w_q = delta_phi.get_gradient(q);
                differential_change_flux +=
                  this->calculate_jacobian_viscous_flux(w_q, grad_w_q, delta_w_q, grad_delta_w_q);
              }
            delta_phi.submit_gradient(residual_rhs_scaling_factor * differential_change_flux, q);
          }

        delta_phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::local_face_jacobian(
    const MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(this->scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 this->comp_flow_dof_idx,
                                                 this->comp_flow_quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(this->scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 this->comp_flow_dof_idx,
                                                 this->comp_flow_quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> delta_phi_m(this->scratch_data.get_matrix_free(),
                                                       true /*is_interior_face*/,
                                                       this->comp_flow_dof_idx,
                                                       this->comp_flow_quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> delta_phi_p(this->scratch_data.get_matrix_free(),
                                                       false /*is_interior_face*/,
                                                       this->comp_flow_dof_idx,
                                                       this->comp_flow_quad_idx);


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(this->solution_history.get_current_solution(),
                              EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(this->solution_history.get_current_solution(),
                              EvaluationFlags::values |
                                ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                   EvaluationFlags::gradients :
                                   EvaluationFlags::nothing));

        delta_phi_p.reinit(face);
        delta_phi_p.gather_evaluate(src,
                                    EvaluationFlags::values |
                                      ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                         EvaluationFlags::gradients :
                                         EvaluationFlags::nothing));

        delta_phi_m.reinit(face);
        delta_phi_m.gather_evaluate(src,
                                    EvaluationFlags::values |
                                      ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                         EvaluationFlags::gradients :
                                         EvaluationFlags::nothing));


        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const std::pair<ConservedVariablesType, ConservedVariablesType> w_q = {
              phi_m.get_value(q), phi_p.get_value(q)};
            const std::pair<ConservedVariablesType, ConservedVariablesType> delta_w_q = {
              delta_phi_m.get_value(q), delta_phi_p.get_value(q)};

            ConservedVariablesGradType numerical_flux =
              this->calculate_jacobian_convective_numerical_flux(w_q,
                                                                 delta_w_q,
                                                                 phi_m.normal_vector(q));

            if (this->comp_flow_data.dynamic_viscosity > 0)
              numerical_flux -= this->calculate_jacobian_viscous_numerical_flux(
                {phi_m.get_value(q), phi_p.get_value(q)},
                {phi_m.get_gradient(q), phi_p.get_gradient(q)},
                {delta_phi_m.get_value(q), delta_phi_p.get_value(q)},
                {delta_phi_m.get_gradient(q), delta_phi_p.get_gradient(q)},
                phi_m.normal_vector(q),
                std::max(phi_m.read_cell_data(this->interior_penalty_parameter),
                         phi_p.read_cell_data(this->interior_penalty_parameter)));
            ConservedVariablesType flux;
            for (unsigned int i = 0; i < dim + 2; ++i)
              {
                flux[i] = numerical_flux[i] * phi_m.normal_vector(q);
              }

            if (this->comp_flow_data.dynamic_viscosity > 0)
              {
                const ConservedVariablesGradType jump =
                  VectorTools::dyadic_product(phi_m.get_value(q) - phi_p.get_value(q),
                                              phi_m.normal_vector(q));
                const ConservedVariablesGradType delta_jump =
                  VectorTools::dyadic_product(delta_phi_m.get_value(q) - delta_phi_p.get_value(q),
                                              phi_m.normal_vector(q));
                const ConservedVariablesGradType grad_flux_m =
                  this->calculate_jacobian_viscous_flux(phi_m.get_value(q),
                                                        jump,
                                                        delta_phi_m.get_value(q),
                                                        delta_jump);
                const ConservedVariablesGradType grad_flux_p =
                  this->calculate_jacobian_viscous_flux(phi_p.get_value(q),
                                                        jump,
                                                        delta_phi_p.get_value(q),
                                                        delta_jump);
                delta_phi_m.submit_gradient(-0.5 * residual_rhs_scaling_factor * grad_flux_m, q);
                delta_phi_p.submit_gradient(-0.5 * residual_rhs_scaling_factor * grad_flux_p, q);
              }
            delta_phi_m.submit_value(residual_rhs_scaling_factor * flux, q);
            delta_phi_p.submit_value(-residual_rhs_scaling_factor * flux, q);
          }

        delta_phi_p.integrate_scatter(EvaluationFlags::values |
                                        ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                           EvaluationFlags::gradients :
                                           EvaluationFlags::nothing),
                                      dst);
        delta_phi_m.integrate_scatter(EvaluationFlags::values |
                                        ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                           EvaluationFlags::gradients :
                                           EvaluationFlags::nothing),
                                      dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorImplicit<dim, number>::local_boundary_face_jacobian(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(this->scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 this->comp_flow_dof_idx,
                                                 this->comp_flow_quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> delta_phi_m(this->scratch_data.get_matrix_free(),
                                                       true /*is_interior_face*/,
                                                       this->comp_flow_dof_idx,
                                                       this->comp_flow_quad_idx);


    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_m.reinit(face);

        phi_m.gather_evaluate(this->solution_history.get_current_solution(),
                              EvaluationFlags::values | EvaluationFlags::gradients);

        delta_phi_m.reinit(face);
        delta_phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);


        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            const auto w_m            = phi_m.get_value(q);
            const auto grad_w_m       = phi_m.get_gradient(q);
            const auto delta_w_m      = delta_phi_m.get_value(q);
            const auto grad_delta_w_m = delta_phi_m.get_gradient(q);
            const auto normal         = phi_m.normal_vector(q);
            const auto boundary_id    = this->scratch_data.get_matrix_free().get_boundary_id(face);

            const auto [w_p, grad_w_p, delta_w_p, grad_delta_w_p] =
              this->get_adjacent_jacobian_face_values_at_boundary(phi_m.quadrature_point(q),
                                                                  normal,
                                                                  boundary_id,
                                                                  w_m,
                                                                  delta_w_m,
                                                                  grad_w_m,
                                                                  grad_delta_w_m);

            ConservedVariablesGradType numerical_flux =
              this->calculate_jacobian_convective_numerical_flux({w_m, w_p},
                                                                 {delta_w_m, delta_w_p},
                                                                 normal);

            if (this->comp_flow_data.dynamic_viscosity > 0)
              numerical_flux -= this->calculate_jacobian_viscous_numerical_flux(
                {w_m, w_p},
                {grad_w_m, grad_w_p},
                {delta_w_m, delta_w_p},
                {grad_delta_w_m, grad_delta_w_p},
                phi_m.normal_vector(q),
                phi_m.read_cell_data(this->interior_penalty_parameter));
            ConservedVariablesType flux;
            for (unsigned int i = 0; i < dim + 2; ++i)
              {
                flux[i] = numerical_flux[i] * phi_m.normal_vector(q);
              }

            if (this->comp_flow_data.dynamic_viscosity > 0)
              {
                const ConservedVariablesGradType jump =
                  VectorTools::dyadic_product(w_m - w_p, phi_m.normal_vector(q));
                const ConservedVariablesGradType delta_jump =
                  VectorTools::dyadic_product(delta_w_m - delta_w_p, phi_m.normal_vector(q));
                const ConservedVariablesGradType grad_flux_m =
                  this->calculate_jacobian_viscous_flux(w_m, jump, delta_w_m, delta_jump);
                delta_phi_m.submit_gradient(-0.5 * residual_rhs_scaling_factor * grad_flux_m, q);
              }
            delta_phi_m.submit_value(residual_rhs_scaling_factor * flux, q);
          }

        delta_phi_m.integrate_scatter(EvaluationFlags::values |
                                        ((this->comp_flow_data.dynamic_viscosity > 0) ?
                                           EvaluationFlags::gradients :
                                           EvaluationFlags::nothing),
                                      dst);
      }
  }

  template class CompressibleFlowOperatorImplicit<1, double>;
  template class CompressibleFlowOperatorImplicit<2, double>;
  template class CompressibleFlowOperatorImplicit<3, double>;
} // namespace MeltPoolDG::Flow
