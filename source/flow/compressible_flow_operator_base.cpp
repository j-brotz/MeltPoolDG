#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/flow/compressible_flow_operator_base.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <meltpooldg/flow/compressible_flow_utils.h>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  CompressibleFlowOperatorBase<dim, number>::CompressibleFlowOperatorBase(
    const CompressibleFlowData                     &compressible_flow_data_in,
    const ScratchData<dim>                         &scratch_data_in,
    ::TimeIntegration::SolutionHistory<VectorType> &solution_history_in,
    const unsigned int                              comp_flow_dof_idx_in,
    const unsigned int                              comp_flow_quad_idx_in)
    : comp_flow_data(compressible_flow_data_in)
    , scratch_data(scratch_data_in)
    , solution_history(solution_history_in)
    , comp_flow_dof_idx(comp_flow_dof_idx_in)
    , comp_flow_quad_idx(comp_flow_quad_idx_in)
    , calculator_functions(comp_flow_data)
  {}

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::reinit()
  {
    calculate_interior_penalty_parameter();
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_body_force(
    std::unique_ptr<dealii::Function<dim>> body_force_in)
  {
    AssertDimension(body_force_in->n_components, dim);
    body_force = std::move(body_force_in);
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_inflow_boundary(
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>> &inflow_bc)
  {
    inflow_boundaries = inflow_bc;
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_subsonic_outflow_with_fixed_static_pressure(
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &outflow_fixed_pressure_bc)
  {
    subsonic_outflow_fixed_pressure = outflow_fixed_pressure_bc;
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_subsonic_outflow_with_fixed_energy(
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &outflow_fixed_energy_bc)
  {
    subsonic_outflow_fixed_energy = outflow_fixed_energy_bc;
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_slip_wall_boundary(
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &slip_wall_bc)
  {
    for (const auto &boundary : slip_wall_bc)
      slip_wall_boundaries.insert(boundary.first);
  }


  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::set_no_slip_adiabatic_wall_boundary(
    const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      &no_slip_wall_bc)
  {
    for (const auto &boundary : no_slip_wall_bc)
      no_slip_adiabatic_wall_boundaries.insert(boundary.first);
  }

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::get_adjacent_face_values_at_boundary(
    const dealii::Point<dim, dealii::VectorizedArray<number>>     &q_point,
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal,
    unsigned int                                                   boundary_id,
    const ConservedVariablesType                                  &w_m,
    ConservedVariablesType                                        &w_p,
    const ConservedVariablesGradType                              &grad_w_m,
    ConservedVariablesGradType                                    &grad_w_p) const
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
  CompressibleFlowOperatorBase<dim, number>::local_apply_inverse_mass_matrix(
    const dealii::MatrixFree<dim, number> &,
    VectorType                          &dst,
    const VectorType                    &src,
    const std::pair<unsigned, unsigned> &cell_range) const
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

  template <int dim, typename number>
  void
  CompressibleFlowOperatorBase<dim, number>::calculate_interior_penalty_parameter()
  {
    if (comp_flow_data.dynamic_viscosity > 0.)
      calculate_penalty_parameter(interior_penalty_parameter,
                                  scratch_data.get_matrix_free(),
                                  comp_flow_data.domain_representation_type,
                                  comp_flow_dof_idx);
  }


  template class CompressibleFlowOperatorBase<1, double>;
  template class CompressibleFlowOperatorBase<2, double>;
  template class CompressibleFlowOperatorBase<3, double>;
} // namespace MeltPoolDG::Flow