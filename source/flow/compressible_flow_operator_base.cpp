#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/flow/compressible_flow_operator_base.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <meltpooldg/flow/compressible_flow_utils.h>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  CompressibleFlowOperatorBase<dim, number>::CompressibleFlowOperatorBase(
    const CompressibleFlowData &compressible_flow_data_in,
    const ScratchData<dim>     &scratch_data_in,
    const unsigned int          comp_flow_dof_idx_in,
    const unsigned int          comp_flow_quad_idx_in)
    : comp_flow_data(compressible_flow_data_in)
    , scratch_data(scratch_data_in)
    , comp_flow_dof_idx(comp_flow_dof_idx_in)
    , comp_flow_quad_idx(comp_flow_quad_idx_in)
  {}

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::reinit()
  {
    // calculate interior penalty parameter
    if (comp_flow_data.dynamic_viscosity != 0.)
      calculate_penalty_parameter(interior_penalty_parameter,
                                  scratch_data.get_matrix_free(),
                                  comp_flow_dof_idx);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_body_force(
    std::unique_ptr<Function<dim>> body_force_in)
  {
    AssertDimension(body_force_in->n_components, dim);
    body_force = std::move(body_force_in);
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_inflow_boundary(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &inflow_bc)
  {
    inflow_boundaries = inflow_bc;
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_subsonic_outflow_with_fixed_static_pressure(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &outflow_fixed_pressure_bc)
  {
    subsonic_outflow_fixed_pressure = outflow_fixed_pressure_bc;
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_subsonic_outflow_with_fixed_energy(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &outflow_fixed_energy_bc)
  {
    subsonic_outflow_fixed_energy = outflow_fixed_energy_bc;
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_slip_wall_boundary(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &slip_wall_bc)
  {
    for (const auto &boundary : slip_wall_bc)
      slip_wall_boundaries.insert(boundary.first);
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_no_slip_adiabatic_wall_boundary(
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> &no_slip_wall_bc)
  {
    for (const auto &boundary : no_slip_wall_bc)
      no_slip_adiabatic_wall_boundaries.insert(boundary.first);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::get_adjacent_face_values_at_boundary(
    const Point<dim, VectorizedArray<number>>     &q_point,
    const Tensor<1, dim, VectorizedArray<number>> &normal,
    unsigned int                                   boundary_id,
    const ConservedVariablesType                  &w_m,
    ConservedVariablesType                        &w_p,
    const ConservedVariablesGradType              &grad_w_m,
    ConservedVariablesGradType                    &grad_w_p) const
  {
    if (slip_wall_boundaries.contains(boundary_id))
      {
        // homogeneous Neumann
        auto rho_u_dot_n = w_m[1] * normal[0];
        for (unsigned int d = 1; d < dim; ++d)
          rho_u_dot_n += w_m[1 + d] * normal[d];
        w_p[0]      = w_m[0];
        grad_w_p[0] = -(grad_w_m[0]);
        // symmetry
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
            grad_w_p[d + 1] =
              grad_w_m[d + 1] - 2. * scalar_product(grad_w_m[d + 1], normal) * normal;
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -(grad_w_m[dim + 1]);
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (no_slip_adiabatic_wall_boundaries.contains(boundary_id))
      {
        // homogeneous Neumann
        w_p[0]      = w_m[0];
        grad_w_p[0] = -(grad_w_m[0]);
        // Dirichlet
        for (unsigned int d = 0; d < dim; ++d)
          {
            w_p[d + 1]      = 0.;
            grad_w_p[d + 1] = grad_w_m[d + 1];
          }
        // homogeneous Neumann
        grad_w_p[dim + 1] = -(grad_w_m[dim + 1]);
        w_p[dim + 1]      = w_m[dim + 1];
      }
    else if (inflow_boundaries.contains(boundary_id))
      {
        // Dirichlet
        w_p = VectorTools::evaluate_function_at_vectorized_points<dim, number, dim + 2>(
          *inflow_boundaries.find(boundary_id)->second, q_point);
        grad_w_p = grad_w_m;
      }
    else if (subsonic_outflow_fixed_pressure.contains(boundary_id))
      {
        // homogeneous Neumann
        w_p      = w_m;
        grad_w_p = -grad_w_m;

        // Dirichlet
        auto p_dyn = VectorizedArray<number>(0.);
        for (unsigned int i = 1; i < dim + 1; ++i)
          p_dyn += w_m[i] * w_m[i];

        p_dyn /= (w_m[0] * 2.);
        w_p[dim + 1] =
          VectorTools::evaluate_function_at_vectorized_points<dim, number>(
            *subsonic_outflow_fixed_pressure.find(boundary_id)->second, q_point, dim + 1) /
            (comp_flow_data.gamma - 1.) +
          p_dyn;
        grad_w_p[dim + 1] = grad_w_m[dim + 1];
      }
    else if (subsonic_outflow_fixed_energy.contains(boundary_id))
      {
        // homogeneous Neumann
        w_p      = w_m;
        grad_w_p = -grad_w_m;

        // Dirichlet
        w_p[dim + 1] = VectorTools::evaluate_function_at_vectorized_points<dim, number>(
          *subsonic_outflow_fixed_energy.find(boundary_id)->second, q_point, dim + 1);
        grad_w_p[dim + 1] = grad_w_m[dim + 1];
      }
    else
      AssertThrow(false,
                  ExcMessage("Unknown boundary id, did "
                             "you set a boundary condition for "
                             "this part of the domain boundary?"));
  }



  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::local_apply_cell(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(scratch_data.get_matrix_free(),
                                               comp_flow_dof_idx,
                                               comp_flow_quad_idx);
    FECellIntegrator<dim, dim + 2, number> fe_user_rhs(scratch_data.get_matrix_free(),
                                                       comp_flow_dof_idx,
                                                       comp_flow_quad_idx);

    Tensor<1, dim, VectorizedArray<number>> constant_body_force;
    const Functions::ConstantFunction<dim> *constant_function =
      dynamic_cast<Functions::ConstantFunction<dim> *>(body_force.get());

    if (constant_function)
      constant_body_force =
        VectorTools::evaluate_function_at_vectorized_points(*constant_function,
                                                            Point<dim, VectorizedArray<number>>());

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              ((comp_flow_data.dynamic_viscosity > 0) ? EvaluationFlags::gradients :
                                                                        EvaluationFlags::nothing));
        fe_user_rhs.reinit(cell);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto w_q = phi.get_value(q);

            auto flux = calculate_convective_flux(w_q);

            if (comp_flow_data.dynamic_viscosity > 0)
              {
                const auto grad_w_q = phi.get_gradient(q);
                flux -= calculate_viscous_flux(w_q, grad_w_q);
              }

            phi.submit_gradient(flux, q);

            dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>> forcing;
            if (body_force.get() != nullptr)
              {
                const Tensor<1, dim, VectorizedArray<number>> force =
                  constant_function ?
                    constant_body_force :
                    VectorTools::evaluate_function_at_vectorized_points(*body_force,
                                                                        phi.quadrature_point(q));

                for (unsigned int d = 0; d < dim; ++d)
                  forcing[d + 1] = w_q[0] * force[d];
                for (unsigned int d = 0; d < dim; ++d)
                  forcing[dim + 1] += force[d] * w_q[d + 1];
                phi.submit_value(forcing, q);
              }
          }

        phi.integrate_scatter(((body_force.get() != nullptr) ? EvaluationFlags::values :
                                                               EvaluationFlags::nothing) |
                                EvaluationFlags::gradients,
                              dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::local_apply_face(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi_m(scratch_data.get_matrix_free(),
                                                 true /*is_interior_face*/,
                                                 comp_flow_dof_idx,
                                                 comp_flow_quad_idx);
    FEFaceIntegrator<dim, dim + 2, number> phi_p(scratch_data.get_matrix_free(),
                                                 false /*is_interior_face*/,
                                                 comp_flow_dof_idx,
                                                 comp_flow_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src,
                              EvaluationFlags::values | ((comp_flow_data.dynamic_viscosity > 0) ?
                                                           EvaluationFlags::gradients :
                                                           EvaluationFlags::nothing));

        phi_m.reinit(face);
        phi_m.gather_evaluate(src,
                              EvaluationFlags::values | ((comp_flow_data.dynamic_viscosity > 0) ?
                                                           EvaluationFlags::gradients :
                                                           EvaluationFlags::nothing));


        for (const unsigned int q : phi_m.quadrature_point_indices())
          {
            auto numerical_flux =
              calculate_convective_numerical_flux(phi_m.get_value(q),
                                                  phi_p.get_value(q),
                                                  phi_m.normal_vector(q),
                                                  comp_flow_data.numerical_flux_type);

            if (comp_flow_data.dynamic_viscosity > 0)
              numerical_flux -= calculate_viscous_numerical_flux(
                phi_m.get_value(q),
                phi_p.get_value(q),
                phi_m.get_gradient(q),
                phi_p.get_gradient(q),
                phi_m.normal_vector(q),
                std::max(phi_m.read_cell_data(interior_penalty_parameter),
                         phi_p.read_cell_data(interior_penalty_parameter)));

            // interior penalty
            if (comp_flow_data.dynamic_viscosity > 0)
              {
                const auto viscous_numerical_flux =
                  calculate_viscous_numerical_flux_gradient(phi_m.get_value(q),
                                                            phi_p.get_value(q),
                                                            phi_m.normal_vector(q));

                // since we approach the face only once, we submit the contributions
                // to the face integral of the two neighbouring elements.
                phi_m.submit_gradient(viscous_numerical_flux.first, q);
                phi_p.submit_gradient(viscous_numerical_flux.second, q);
              }
            phi_m.submit_value(-numerical_flux, q);
            phi_p.submit_value(numerical_flux, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values | ((comp_flow_data.dynamic_viscosity > 0) ?
                                                             EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing),
                                dst);
        phi_m.integrate_scatter(EvaluationFlags::values | ((comp_flow_data.dynamic_viscosity > 0) ?
                                                             EvaluationFlags::gradients :
                                                             EvaluationFlags::nothing),
                                dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::local_apply_boundary_face(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &face_range) const
  {
    FEFaceIntegrator<dim, dim + 2, number> phi(scratch_data.get_matrix_free(),
                                               true,
                                               comp_flow_dof_idx,
                                               comp_flow_quad_idx);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : phi.quadrature_point_indices())
          {
            const auto w_m      = phi.get_value(q);
            const auto normal   = phi.normal_vector(q);
            const auto grad_w_m = phi.get_gradient(q);

            Tensor<1, dim + 2, VectorizedArray<number>>                 w_p;
            Tensor<1, dim + 2, Tensor<1, dim, VectorizedArray<number>>> grad_w_p;

            const auto boundary_id = scratch_data.get_matrix_free().get_boundary_id(face);

            // TODO
            get_adjacent_face_values_at_boundary(
              phi.quadrature_point(q), normal, boundary_id, w_m, w_p, grad_w_m, grad_w_p);

            auto flux = calculate_convective_numerical_flux(w_m,
                                                            w_p,
                                                            normal,
                                                            comp_flow_data.numerical_flux_type);

            if (comp_flow_data.dynamic_viscosity > 0)
              flux -=
                calculate_viscous_numerical_flux(w_m,
                                                 w_p,
                                                 grad_w_m,
                                                 grad_w_p,
                                                 phi.normal_vector(q),
                                                 phi.read_cell_data(interior_penalty_parameter));

            phi.submit_value(-flux, q);

            if (comp_flow_data.dynamic_viscosity > 0)
              {
                auto numerical_flux_gradient =
                  calculate_viscous_numerical_flux_gradient(w_m, w_p, phi.normal_vector(q)).first;

                phi.submit_gradient(numerical_flux_gradient, q);
              }
          }

        phi.integrate_scatter(EvaluationFlags::values | ((comp_flow_data.dynamic_viscosity > 0) ?
                                                           EvaluationFlags::gradients :
                                                           EvaluationFlags::nothing),
                              dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, number> &,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned, unsigned>              &cell_range) const
  {
    FECellIntegrator<dim, dim + 2, number> phi(scratch_data.get_matrix_free(),
                                               comp_flow_dof_idx,
                                               comp_flow_dof_idx);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, number> inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
      }
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::update_boundary_conditions(const number time) const
  {
    for (auto &i : this->inflow_boundaries)
      i.second->set_time(time);
    for (auto &i : this->subsonic_outflow_fixed_pressure)
      i.second->set_time(time);
    for (auto &i : this->subsonic_outflow_fixed_energy)
      i.second->set_time(time);
  }

  template class CompressibleFlowOperatorBase<1, double>;
  template class CompressibleFlowOperatorBase<2, double>;
  template class CompressibleFlowOperatorBase<3, double>;
} // namespace MeltPoolDG::Flow