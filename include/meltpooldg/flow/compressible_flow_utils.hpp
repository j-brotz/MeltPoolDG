#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/flow/compressible_flow_boundary_conditions.hpp>
#include <meltpooldg/flow/compressible_flow_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

namespace MeltPoolDG::Flow
{
  struct CompressibleFlowData;
  /**
   * Struct providing type aliases that might be useful in the compressible flow implementations.
   */
  struct CompressibleFlowTypes
  {
    template <int dim, typename number>
    /**
     * Type of the conserved variables []
     */
    using ConservedVariablesType = dealii::Tensor<1, dim + 2, dealii::VectorizedArray<number>>;

    /**
     * Type of the gradient of the conserved variables []
     */
    template <int dim, typename number>
    using ConservedVariablesGradType =
      dealii::Tensor<1, dim + 2, dealii::Tensor<1, dim, dealii::VectorizedArray<number>>>;
  };

  /**
   * Calculate the velocity from the conserved variables by computing u = (ρu)/ρ.
   *
   * @param conserved_variables Current values of the conserved variables.
   *
   * @return Current velocity.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_velocity(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables);

  /**
   * Calculate the gradient of the  velocity from the conserved variables and their gradients by
   * computing grad(u) = 1/ρ * (grad(ρu) - u*grad(ρ)).
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param grad_conserved_variables Current gradient of the conserved variables.
   *
   * @return Current gradient of the velocity.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    calculate_grad_velocity(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
        &grad_conserved_variables);

  /**
   * Calculate the gradient of the temperature from the conserved variables and their gradients by
   * computing grad(T) = (γ-1)/R * (grad(E) - grad(u)*u).
   *
   * @param conserved_variables Current values of the conserved variables.
   * @param grad_conserved_variables Current gradient of the conserved variables.
   * @param gamma Heat capacity ratio of the flow field.
   * @param specific_gas_constant Specific gas constant of the flow field.
   *
   * @return Current gradient of the temperature field.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_grad_T(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
            &grad_conserved_variables,
      number gamma,
      number specific_gas_constant);

  /**
   * Contracts the given tensor of the gradient of conserved variables with the given normal
   * vector.
   *
   * @param grad Current gradient of the conserved variables.
   * @param normal Normal vector.
   *
   * @return Result of the contraction, i.e. grad*normal.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    CompressibleFlowTypes::ConservedVariablesType<dim, number>
    contract_tensor_with_normal(
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number> &grad,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>        &normal);

  /**
   * Computes the average of the passed tensors and contracts the corresponding result with the
   * given normal vector. In other words the function computes 0.5*(grad_m+grad_p)*normal.
   *
   * @param grad_m Current gradient of the conserved variables on the inner face.
   * @param grad_p Current gradient of the conserved variables on the outer face.
   * @param normal Normal vector.
   *
   * @return Result of the contraction.
   */
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    CompressibleFlowTypes::ConservedVariablesType<dim, number>
    contract_average_tensor_with_normal(
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number> &grad_m,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number> &grad_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>        &normal);

  /**
   * This function computes the local values of the internal penalty parameter used in the viscous
   * numerical flux.
   *
   * @param array_penalty_parameter Array in which the values of the penalty parameter are stored.
   * @param matrix_free Matrix-free object providing the required geometrical data.
   * @param domain_representation_type Numerical operator type (cut or fitted_mesh).
   * @param dof_index Index of the relevant dof handler in the matrix-free object.
   * @param scaling_factor Additional scaling factor to scale the penalty parameter.
   */
  template <int dim, typename Number>
  void
  calculate_penalty_parameter(AlignedVector<VectorizedArray<Number>> &array_penalty_parameter,
                              const MatrixFree<dim, Number>          &matrix_free,
                              const std::string                      &domain_representation_type,
                              const unsigned int                      dof_index      = 0,
                              const double                            scaling_factor = 1.0);

  /**
   * A struct providing the relevant data required by all compressible flow solvers.
   */
  template <int dim, typename number>
  struct CompressibleFlowScratchData
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor
     *
     * @param flow_data_in Reference to the flow data object.
     * @param scratch_data_in Reference to the scratch data object.
     * @param dof_idx_in Relevant dof index of the flow solver in the scratch data object.
     * @param quad_idx_in Relevant quadrature index of the flow solver in the scratch data object.
     */
    CompressibleFlowScratchData(const CompressibleFlowData &flow_data_in,
                                const ScratchData<dim>     &scratch_data_in,
                                const unsigned int          dof_idx_in,
                                const unsigned int          quad_idx_in)
      : flow_data(flow_data_in)
      , scratch_data(scratch_data_in)
      , dof_idx(dof_idx_in)
      , quad_idx(quad_idx_in)
    {}

    /**
     * Set up the internal data structures, i.e. allocate memory for the solution history object and
     * precompute the penalty parameter for the symmetric interior penalty method.
     *
     * @param solution_history_size Size of the solution history object, i.e. the number of vectors at different
     * concrete times n for which the solution history is responsible.
     */
    void
    reinit(const unsigned solution_history_size)
    {
      solution_history.resize(solution_history_size);
      solution_history.apply(
        [&scratch_data = scratch_data, comp_flow_dof_idx = dof_idx](VectorType &v) {
          scratch_data.initialize_dof_vector(v, comp_flow_dof_idx);
        });

      if (flow_data.dynamic_viscosity > 0.)
        calculate_penalty_parameter(interior_penalty_parameter,
                                    scratch_data.get_matrix_free(),
                                    flow_data.domain_representation_type,
                                    dof_idx);
    }

    const CompressibleFlowData flow_data;

    const ScratchData<dim> &scratch_data;

    const unsigned int dof_idx  = 0;
    const unsigned int quad_idx = 0;

    CompressibleFlowBoundaryConditions<dim, number> boundary_conditions;

    AlignedVector<VectorizedArray<number>> interior_penalty_parameter;

    ::TimeIntegration::SolutionHistory<VectorType> solution_history;

    std::unique_ptr<dealii::Function<dim>> body_force;
  };

  /********************************************************************************************
   * Inlined function definitions
   * *************************************************************************************+****/
  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_velocity(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables)
  {
    const dealii::VectorizedArray<number> inverse_density =
      dealii::VectorizedArray<number>(1.) / conserved_variables[0];

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity;
    for (unsigned int d = 0; d < dim; ++d)
      velocity[d] = conserved_variables[1 + d] * inverse_density;

    return velocity;
  }


  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<2, dim, dealii::VectorizedArray<number>>
    calculate_grad_velocity(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
        &grad_conserved_variables)
  {
    const dealii::VectorizedArray<number> inverse_density =
      dealii::VectorizedArray<number>(1.) / conserved_variables[0];
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> velocity =
      calculate_velocity<dim, number>(conserved_variables);

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_rho;
    for (unsigned int d = 0; d < dim; ++d)
      grad_rho[d] = grad_conserved_variables[0][d];

    dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_rho_velocity;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        grad_rho_velocity[d][e] = grad_conserved_variables[1 + d][e];

    dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_velocity;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        grad_velocity[d][e] =
          inverse_density * (grad_rho_velocity[d][e] - velocity[d] * grad_rho[e]);

    return grad_velocity;
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
    calculate_grad_T(
      const CompressibleFlowTypes::ConservedVariablesType<dim, number> &conserved_variables,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number>
                  &grad_conserved_variables,
      const number gamma,
      const number specific_gas_constant)
  {
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> u =
      calculate_velocity<dim, number>(conserved_variables);
    const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> grad_u =
      calculate_grad_velocity<dim, number>(conserved_variables, grad_conserved_variables);
    const dealii::VectorizedArray<number> rho     = conserved_variables[0];
    const dealii::VectorizedArray<number> inv_rho = dealii::VectorizedArray<number>(1.) / rho;
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_rho =
      grad_conserved_variables[0];
    const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> grad_E =
      inv_rho *
      (grad_conserved_variables[dim + 1] - inv_rho * conserved_variables[dim + 1] * grad_rho);
    return (gamma - 1.0) / specific_gas_constant * (grad_E - grad_u * u);
  }

  template <int dim, typename number>
  inline DEAL_II_ALWAYS_INLINE //
    CompressibleFlowTypes::ConservedVariablesType<dim, number>
    contract_tensor_with_normal(
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number> &grad,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>        &normal)
  {
    CompressibleFlowTypes::ConservedVariablesType<dim, number> result;

    result[0] = grad[0] * normal;

    for (unsigned int e = 0; e < dim; ++e)
      result[e + 1] = grad[e + 1] * normal;

    result[dim + 1] = grad[dim + 1] * normal;

    return result;
  }

  template <int dim, typename number>
  DEAL_II_ALWAYS_INLINE //
    CompressibleFlowTypes::ConservedVariablesType<dim, number>
    contract_average_tensor_with_normal(
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number> &grad_m,
      const CompressibleFlowTypes::ConservedVariablesGradType<dim, number> &grad_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>>        &normal)
  {
    CompressibleFlowTypes::ConservedVariablesType<dim, number> result;

    result[0] = (grad_m[0] + grad_p[0]) * normal;

    for (unsigned int e = 0; e < dim; ++e)
      result[e + 1] = (grad_m[e + 1] + grad_p[e + 1]) * normal;

    result[dim + 1] = (grad_m[dim + 1] + grad_p[dim + 1]) * normal;

    return 0.5 * result;
  }

  template <int dim, typename Number>
  void
  calculate_penalty_parameter(AlignedVector<VectorizedArray<Number>> &array_penalty_parameter,
                              const MatrixFree<dim, Number>          &matrix_free,
                              const std::string                      &domain_representation_type,
                              const unsigned int                      dof_index,
                              const double                            scaling_factor)
  {
    const unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);

    Mapping<dim> const       &mapping = *matrix_free.get_mapping_info().mapping;
    FiniteElement<dim> const &fe      = matrix_free.get_dof_handler(dof_index).get_fe();
    unsigned int const        degree  = fe.degree;

    // use penalty factor for hypercube elements according to K. Hillewaert, Development of the
    // discontinuous Galerkin method for high-resolution, large scale CFD and acoustics in
    // industrial geometries, PhD thesis, Univ. de Louvain, 2013.
    const double fac = scaling_factor * (degree + 1.0) * (degree + 1.0);

    auto const reference_cells =
      matrix_free.get_dof_handler(dof_index).get_triangulation().get_reference_cells();
    AssertThrow(reference_cells.size() == 1, ExcMessage("No mixed meshes allowed."));

    auto const quadrature = reference_cells[0].template get_gauss_type_quadrature<dim>(degree + 1);
    FEValues<dim> fe_values(mapping, fe, quadrature, update_JxW_values);

    auto const face_quadrature =
      reference_cells[0].face_reference_cell(0).template get_gauss_type_quadrature<dim - 1>(degree +
                                                                                            1);
    FEFaceValues<dim> fe_face_values(mapping, fe, face_quadrature, update_JxW_values);

    if (domain_representation_type == "fitted")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);
                fe_values.reinit(cell);

                // calculate cell volume
                Number volume = 0;
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    volume += fe_values.JxW(q);
                  }

                // calculate surface area
                Number surface_area = 0;
                for (unsigned int const f : cell->face_indices())
                  {
                    fe_face_values.reinit(cell, f);
                    Number const factor =
                      (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ? 1. : 0.5;
                    for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                      {
                        surface_area += fe_face_values.JxW(q) * factor;
                      }
                  }

                array_penalty_parameter[i][v] = surface_area / volume * fac;
              }
          }
      }
    else if (domain_representation_type == "cut")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);

                // simplified computation for hypercube elements
                array_penalty_parameter[i][v] = fac / cell->minimum_vertex_distance();
              }
          }
      }
    else
      AssertThrow(false,
                  dealii::ExcMessage("The domain representation type '" +
                                     domain_representation_type + "' is not supported."));
  }
} // namespace MeltPoolDG::Flow
