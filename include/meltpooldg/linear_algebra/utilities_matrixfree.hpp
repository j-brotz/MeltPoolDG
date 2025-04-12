#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/core/operator_base.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG::Utilities::MatrixFree
{
  template <typename number = double>
  using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
  /**
   * Compute the modified right-handside for (inhomogeneous) dirichlet boundary
   * conditions x_d
   *
   * A * x = B
   *
   * We actually solve
   *
   * A * x_0 = b - A * x_d
   *
   * with zero Dirichlet boundary conditions.
   *
   * @note When using this function, it must be ensured that this->dof_idx is
   *       used for reading the source vector "x".
   */
  template <int dim,
            typename number           = double,
            typename DoFVectorType    = VectorType<number>,
            typename SrcRhsVectorType = VectorType<number>>
  inline void
  create_rhs_and_apply_dirichlet_matrixfree(
    OperatorMatrixFree<dim, number>     &operator_base,
    DoFVectorType                       &rhs,
    const SrcRhsVectorType              &src,
    const ScratchData<dim, dim, number> &scratch_data,
    const unsigned int                   dof_idx,
    const unsigned int                   dof_no_bc_idx,
    const bool                           zero_out,
    const std::optional<std::pair<std::vector<unsigned int>, std::vector<number>>>
      &additional_inhomogeneous_constraints = std::nullopt)
  {
    // The dof index that is used for the DoF Vector from the matrix-vector
    // product must be switched to the one without Dirichlet boundary
    // conditions, such that inhomogeneities are not zeroed out during
    // read_dof_values().
    operator_base.reset_dof_index(dof_no_bc_idx);

    DoFVectorType temp_rhs;
    scratch_data.initialize_dof_vector(temp_rhs, dof_no_bc_idx);

    // This copy is necessary since we observed problems with incompatible
    // communication pattern when using periodic BC.
    SrcRhsVectorType temp_src;
    scratch_data.initialize_dof_vector(temp_src, dof_no_bc_idx);
    temp_src = src;

    DoFVectorType bc_values;
    scratch_data.initialize_dof_vector(bc_values, dof_no_bc_idx);

    // set inhomogeneity
    if (additional_inhomogeneous_constraints)
      {
        const auto &bc = *additional_inhomogeneous_constraints;
        for (unsigned int i = 0; i < bc.first.size(); ++i)
          bc_values.local_element(bc.first[i]) = bc.second[i];
      }

    scratch_data.get_constraint(dof_idx).distribute(bc_values);

    // perform matrix-vector multiplication (with unconstrained system and constrained set in
    // Vector)
    operator_base.vmult(temp_rhs, bc_values);

    // Modify right-hand side
    temp_rhs *= -1.0;
    operator_base.create_rhs(temp_rhs, temp_src);
    /*
     * Zero-out constrained values
     */
    if (zero_out)
      rhs = temp_rhs;
    else
      rhs += temp_rhs;

    // zero-out values of additional inhomogeneity
    if (additional_inhomogeneous_constraints)
      {
        const auto &bc = *additional_inhomogeneous_constraints;
        for (const auto &i : bc.first)
          {
            rhs.local_element(i) = 0;
          }
      }

    scratch_data.get_constraint(dof_idx).set_zero(rhs);

    operator_base.reset_dof_index(dof_idx);
  }

  /**
   * Apply the inverse of the mass matrix to the given dof vector.
   *
   * @param matrix_free Matrix free object on which the applier works on.
   * @param dst Destination vector where the solution is stored.
   * @param src Current solution of the primary variables.
   * @param cell_range Cell range on which the inverse mass matrix is applied.
   * @param dof_idx Relevant dof index in the matrix free object.
   * @param quad_idx Relevant quadrature index in the matrix free object.
   */
  template <int dim, typename number, int n_components>
  void
  local_apply_inverse_mass_matrix(const dealii::MatrixFree<dim, number>       &matrix_free,
                                  VectorType<number>                          &dst,
                                  const VectorType<number>                    &src,
                                  const std::pair<unsigned int, unsigned int> &cell_range,
                                  const unsigned int                           dof_idx,
                                  const unsigned int                           quad_idx)
  {
    dealii::FECellIntegrator<dim, n_components, number> phi(matrix_free, dof_idx, quad_idx);
    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
      phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(dst);
      }
  }
} // namespace MeltPoolDG::Utilities::MatrixFree
