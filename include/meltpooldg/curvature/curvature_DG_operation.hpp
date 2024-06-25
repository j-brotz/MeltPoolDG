/* ---------------------------------------------------------------------
 *
 * Author: Johannes Resch, TUM, May 2024
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/curvature/curvature_data.hpp>
#include <meltpooldg/curvature/curvature_operation_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/normal_vector/helmholtz_DG_operator.hpp>
#include <meltpooldg/utilities/solution_history.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;
  template <int dim, typename Number = double>
  class CurvatureDGOperation
  {
  private:
    using VectorType       = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
    using SparseMatrixType = TrilinosWrappers::SparseMatrix;

  public:
    CurvatureDGOperation(const ScratchData<dim>      &scratch_data_in,
                         const unsigned int           curvature_dof_idx_in,
                         const unsigned int           curvature_quad_idx_in,
                         const BlockVectorType       &solution_normal_vector_in,
                         const CurvatureData<double> &curvature_data);

    void
    solve();

    const LinearAlgebra::distributed::Vector<double> &
    get_curvature() const;

    LinearAlgebra::distributed::Vector<double> &
    get_curvature();

    void
    reinit();

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

  private:
    const ScratchData<dim>     &scratch_data;
    const BlockVectorType      &solution_normal_vector;
    const CurvatureData<double> curvature_data;

    TimeIntegration::SolutionHistory<VectorType> solution_history;

    /*
     *  Based on the following indices the correct DoFHandler or quadrature rule from
     *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
     *  multiple DoFHandlers, quadrature rules, etc.
     */
    const unsigned int curvature_dof_idx;
    const unsigned int curvature_quad_idx;

    const HelmholtzDGOperator<dim, Number> helmholtz_operator;

    /**
     * Applies the domain integral of the right hand side in the form -(w, ∇ o n_i)
     *                                                                             Ω
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param cell_range
     */
    void
    right_hand_side_domain(const MatrixFree<dim, Number>               &data,
                           VectorType                                  &dst,
                           const BlockVectorType                       &src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * Applies the face integral of the right hand side
     * In contrast to the normal operation, the right hand side is not rewritten in divergence form
     * and Gauss divergence theorem is not applied. Therefore this function is empty. We found it to
     * cause less oscillations near singularities.
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param face_range
     */
    template <uint direction>
    void
    right_hand_side_inner_face(
      [[maybe_unused]] const MatrixFree<dim, Number>               &data,
      [[maybe_unused]] VectorType                                  &dst,
      [[maybe_unused]] const BlockVectorType                       &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int> &face_range) const
    {}

    /**
     * Applies the boundary integral of the right hand side
     * In contrast to the normal operation, the right hand side is not rewritten in divergence form
     * and Gauss divergence theorem is not applied. Therefore this function is empty. We found it to
     * cause less oscillations near singularities.
     * @param data the matrix free object
     * @param dst destination where the result is stored
     * @param src source vector
     * @param face_range
     */
    template <uint direction>
    void
    right_hand_side_boundary_face(
      [[maybe_unused]] const MatrixFree<dim, Number>               &data,
      [[maybe_unused]] VectorType                                  &dst,
      [[maybe_unused]] const BlockVectorType                       &src,
      [[maybe_unused]] const std::pair<unsigned int, unsigned int> &face_range) const
    {}
  };
} // namespace MeltPoolDG::LevelSet
