/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Utilities::MatrixFree
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;
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
            typename DoFVectorType    = VectorType,
            typename SrcRhsVectorType = VectorType>
  inline void
  create_rhs_and_apply_dirichlet_matrixfree(
    OperatorBase<dim, number> &operator_base,
    DoFVectorType             &rhs,
    const SrcRhsVectorType    &src,
    const ScratchData<dim>    &scratch_data,
    const unsigned int         dof_idx,
    const unsigned int         dof_no_bc_idx,
    const bool                 zero_out,
    const std::optional<std::pair<std::vector<unsigned int>, std::vector<double>>>
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

    /*
     * perform matrix-vector multiplication (with unconstrained system and constrained set in
     * Vector)
     */
    operator_base.vmult(temp_rhs, bc_values);
    /*
     * Modify right-hand side
     */
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
} // namespace MeltPoolDG::Utilities::MatrixFree
