#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <memory>
#include <utility>


namespace MeltPoolDG::LevelSet
{
  /**
   * This function applies the weak DG form of the Helmholtz operator n_ϕ  - η_n*Δn_ϕ on a source
   * vector. It is used for both the normal vector calculation and the calculation of the
   * interface curvature.
   */
  template <int dim, typename number>
  class HelmholtzDGOperator
  {
  private:
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using SparseMatrixType    = dealii::TrilinosWrappers::SparseMatrix;
    using SparsityPatternType = dealii::TrilinosWrappers::SparsityPattern;

  public:
    HelmholtzDGOperator(const ScratchData<dim, dim, number> &scratch_data_in,
                        const unsigned int                   dof_idx_in,
                        const unsigned int                   quad_idx_in,
                        const number                         filter_parameter_in,
                        const number                         interior_penalty_parameter_in,
                        const PreconditionerType             preconditioner_type_in);
    /**
     * Applies the operator to @p src and safes the result in @p dst
     * @param dst destination vector
     *  @param src source vector
     */
    void
    vmult(dealii::LinearAlgebra::distributed::Vector<number>       &dst,
          const dealii::LinearAlgebra::distributed::Vector<number> &src) const;

    void
    reinit() const;

    const SparseMatrixType &
    get_system_matrix() const
    {
      return this->system_matrix;
    }

    const Preconditioner<dim, VectorType, number> &
    get_preconditioner() const
    {
      return this->preconditioner;
    }

  private:
    const ScratchData<dim, dim, number> &scratch_data;

    const unsigned int dof_idx;
    const unsigned int quad_idx;

    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> damping;
    mutable dealii::AlignedVector<dealii::VectorizedArray<number>> array_penalty_parameter;

    const number filter_parameter;
    const number interior_penalty_parameter;

    mutable SparseMatrixType system_matrix;

    mutable Preconditioner<dim, VectorType, number> preconditioner;

    const PreconditionerType preconditioner_type;


    /**
     * Applies the domain integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    local_apply_domain(const dealii::MatrixFree<dim, number>       &data,
                       VectorType                                  &dst,
                       const VectorType                            &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the inner face integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param face_range
     */
    void
    local_apply_inner_face(const dealii::MatrixFree<dim, number>       &data,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * Applies the boundary face integral
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param face_range
     */
    void
    local_apply_boundary_face(const dealii::MatrixFree<dim, number>       &data,
                              VectorType                                  &dst,
                              const VectorType                            &src,
                              const std::pair<unsigned int, unsigned int> &face_range) const;

    /**
     * The following functions are similar to the ones above, but are need to obtain the matrix
     * from a matrix free implementation
     */

    /**
     * @param cell_integrator
     */
    void
    do_cell_integral_local(dealii::FECellIntegrator<dim, 1, number> &cell_integrator) const;

    /**
     * @param face_integrator_minus
     * @param face_integrator_plus
     */
    void
    do_face_integral_local(dealii::FEFaceIntegrator<dim, 1, number> &face_integrator_minus,
                           dealii::FEFaceIntegrator<dim, 1, number> &face_integrator_plus) const;

    /**
     * @param face_integrator_minus
     */
    void
    do_bounary_integral_local(
      dealii::FEFaceIntegrator<dim, 1, number> &face_integrator_minus) const;

    /**
     * Builds the matrix of the operator
     */
    void
    build_matrix() const;
  };
} // namespace  MeltPoolDG::LevelSet
