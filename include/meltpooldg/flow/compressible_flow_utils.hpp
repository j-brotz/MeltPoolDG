/**
 * @brief Type definitions and helper functions for the compressible flow implementations.
 */

#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

namespace MeltPoolDG::Flow
{
  /**
   * Index set for the components of the compressible Navier-Stokes equations.
   */
  BETTER_ENUM(Idx1D, char, density, momentum_x, energy);
  BETTER_ENUM(Idx2D, char, density, momentum_x, momentum_y, energy);
  BETTER_ENUM(Idx3D, char, density, momentum_x, momentum_y, momentum_z, energy);

  /**
   * Struct providing type aliases that might be useful in the compressible flow implementations.
   */
  struct CompressibleFlowTypes
  {
    /**
     * Type of the conserved variables []
     */
    template <int dim, typename number>
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
  calculate_penalty_parameter(
    dealii::AlignedVector<dealii::VectorizedArray<Number>> &array_penalty_parameter,
    const dealii::MatrixFree<dim, Number>                  &matrix_free,
    const std::string                                      &domain_representation_type,
    const unsigned int                                      dof_index      = 0,
    const Number                                            scaling_factor = 1.0);

  /**
   * @brief An abstract interface for defining external forces acting on the fluid that must be
   * evaluated and incorporated during the cell loop of an explicit time integration scheme.
   *
   * This struct serves as a base for user-defined external fluid force models. Any derived class
   * must implement the two core functions:
   * - @p cell_operation(): Invoked once per cell batch to perform any necessary precomputations.
   * - @p quad_operation(): Invoked at each quadrature point to compute the contribution of the
   *   external force.
   */
  template <int dim, typename number>
  struct ExternalFluidForcesRightHandSideContribution
  {
    virtual ~ExternalFluidForcesRightHandSideContribution() = default;

    /**
     * @brief Function called once per cell batch during the cell loop.
     *
     * @param matrix_free MatrixFree object providing access to the degrees of freedom and geometry.
     * @param cell_batch_id Index of the current cell batch.
     * @param n_lanes Number of lanes (i.e., cells in the batch).
     */
    virtual void
    cell_operation(const dealii::MatrixFree<dim, number> &matrix_free,
                   unsigned int                           cell_batch_id,
                   unsigned int n_lanes = dealii::VectorizedArray<number>::size) = 0;

    /**
     * @brief Function called once per batch of quadrature points to compute the external force
     * contribution at each point.
     *
     * @param q_point Coordinates of the quadrature points.
     * @param w_q Conserved variables at the corresponding quadrature points.
     *
     * @return The computed contribution of the external force to be added to the conservation
     * equations.
     */
    virtual CompressibleFlowTypes::ConservedVariablesType<dim, number>
    quad_operation(const dealii::Point<dim, dealii::VectorizedArray<number>>        &q_point,
                   const CompressibleFlowTypes::ConservedVariablesType<dim, number> &w_q) = 0;
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

  template <int dim, typename Number>
  void
  calculate_penalty_parameter(
    dealii::AlignedVector<dealii::VectorizedArray<Number>> &array_penalty_parameter,
    const dealii::MatrixFree<dim, Number>                  &matrix_free,
    const std::string                                      &domain_representation_type,
    const unsigned int                                      dof_index,
    const Number                                            scaling_factor)
  {
    const unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);

    dealii::Mapping<dim> const       &mapping = *matrix_free.get_mapping_info().mapping;
    dealii::FiniteElement<dim> const &fe      = matrix_free.get_dof_handler(dof_index).get_fe();
    unsigned int const                degree  = fe.degree;

    // use penalty factor for hypercube elements according to K. Hillewaert, Development of the
    // discontinuous Galerkin method for high-resolution, large scale CFD and acoustics in
    // industrial geometries, PhD thesis, Univ. de Louvain, 2013.
    const Number fac = scaling_factor * (degree + 1.0) * (degree + 1.0);

    auto const reference_cells =
      matrix_free.get_dof_handler(dof_index).get_triangulation().get_reference_cells();
    AssertThrow(reference_cells.size() == 1, dealii::ExcMessage("No mixed meshes allowed."));

    auto const quadrature = reference_cells[0].template get_gauss_type_quadrature<dim>(degree + 1);
    dealii::FEValues<dim> fe_values(mapping, fe, quadrature, dealii::update_JxW_values);

    auto const face_quadrature =
      reference_cells[0].face_reference_cell(0).template get_gauss_type_quadrature<dim - 1>(degree +
                                                                                            1);
    dealii::FEFaceValues<dim> fe_face_values(mapping,
                                             fe,
                                             face_quadrature,
                                             dealii::update_JxW_values);

    if (domain_representation_type == "fitted")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename dealii::DoFHandler<dim>::cell_iterator cell =
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
                typename dealii::DoFHandler<dim>::cell_iterator cell =
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
