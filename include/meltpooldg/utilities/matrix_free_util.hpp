#pragma once

#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG
{

  template <int dim, typename Number>
  struct MatrixFreeContext
  {
    const dealii::MatrixFree<dim, Number> &mf;
    unsigned int                           dof_idx;
    unsigned int                           quad_idx;
  };

  /**
   * Helper function that returns the values at a given quadrature point (specified by `q_index`)
   * using the provided finite element evaluation object (`fe_eval`) and ensures that the result
   * is always returned as a `dealii::Tensor`.
   *
   * In deal.II, `FEValues` and similar objects return a `VectorizedArray` directly when there is
   * only a single component. However, the algorithms in this codebase are written to operate on
   * tensors for both 2D and 3D problems and for systems with multiple components. This function
   * provides a uniform interface by converting single-component vectorized values into a
   * `Tensor<1,1,VectorizedArray>` while leaving multi-component values as-is.
   *
   * @param fe_eval The finite element evaluation object.
   * @param q_index Index of the quadrature point.

   * @return Value at the specified quadrature point as a `dealii::Tensor`.
   */
  template <typename FeEval>
  dealii::Tensor<1, FeEval::n_components, dealii::VectorizedArray<typename FeEval::number_type>>
  fe_evaluation_tensor_value_at_q(const FeEval &fe_eval, const unsigned q_index)
  {
    using ValueType           = typename FeEval::value_type;
    using VectorizedArrayType = dealii::VectorizedArray<typename FeEval::number_type>;

    if constexpr (std::is_same_v<ValueType, VectorizedArrayType>)
      {
        dealii::Tensor<1, 1, VectorizedArrayType> t;
        t[0] = fe_eval.get_value(q_index);
        return t;
      }
    else
      return fe_eval.get_value(q_index);
  }

  /**
   * This function returns the cell iterators corresponding to the active SIMD
   * lanes of the specified cell batch.
   *
   * @param mf The matrix-free object defining the cell batch of interest.
   * @param cell_batch_id Index of the cell batch.
   */
  template <int dim, typename number>
  boost::container::small_vector<dealii::TriaIterator<dealii::CellAccessor<dim>>,
                                 dealii::VectorizedArray<number>::size()>
  cells_in_cell_batch(const dealii::MatrixFree<dim, number> &mf, const unsigned int cell_batch_id)
  {
    unsigned int n_active_lanes = mf.n_active_entries_per_cell_batch(cell_batch_id);

    boost::container::small_vector<dealii::TriaIterator<dealii::CellAccessor<dim>>,
                                   dealii::VectorizedArray<number>::size()>
      cells;

    for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
      cells.push_back(mf.get_cell_iterator(cell_batch_id, lane));

    return cells;
  }

  /// An enum of possible matrix representation types that can be computed from a matrix free
  /// object with the below defined functions.
  enum MatrixRepresentationType
  {
    DiagonalMatrix,
    SystemMatrix
  };

  /**
   * A concept setting the requirements on a class for which the matrix representation of the
   * jacobian shall be computed with the below defined function
   * compute_jacobian_matrix_representation().
   */
  template <typename OperatorType, int dim, typename number>
  concept HasJacobianQuadraturePointKernels =
    requires(OperatorType                                                 op,
             dealii::FEEvaluation<dim, -1, 0, dim + 2, number>           &fe_evaluator,
             const dealii::FEEvaluation<dim, -1, 0, dim + 2, number>     &const_fe_evaluator,
             dealii::FEFaceEvaluation<dim, -1, 0, dim + 2, number>       &fe_face_evaluator,
             const dealii::FEFaceEvaluation<dim, -1, 0, dim + 2, number> &const_fe_face_evaluator,
             unsigned int                                                 q_index,
             unsigned int                                                 cell_batch_id) {
      /**
       * Compute the operations for computing the jacobian on the quadrature points for the cell
       * loop.
       */
      op.local_cell_jacobian_kernel(fe_evaluator, const_fe_evaluator, q_index, cell_batch_id);

      /**
       * Compute the operations for computing the jacobian on the quadrature points for the face
       * loop.
       */
      op.local_face_jacobian_kernel(fe_face_evaluator,
                                    fe_face_evaluator,
                                    const_fe_face_evaluator,
                                    const_fe_face_evaluator,
                                    q_index);

      /**
       * Compute the operations for computing the jacobian on the quadrature points for the boundary
       * face loop.
       */
      op.local_boundary_face_jacobian_kernel(fe_face_evaluator, const_fe_face_evaluator, q_index);
    };

  /**
   * Given a matrix-free object, computes the matrix based representation of a Jacobian.
   *
   * Given a matrix free implementation of the jacobian matrix, this function computes the matrix
   * based representation of the jacobian in various forms (diagonal matrix, sparse matrix).
   *
   * @param implicit_operator Operator containing the kernel functions defining the operations at
   * quadrature points.
   * @param dst Destination in which the matrix representation is stored.
   * @param type Type of the desired matrix representation.
   * @param current_solution Solution of the primary variables.
   * @param matrix_free Matrix free object to be worked on.
   * @param dof_idx Relevant dof index in the matrix free object.
   * @param quad_idx Relevant quadrature index in the matrix free object.
   */
  template <int dim,
            int n_components,
            typename number,
            HasJacobianQuadraturePointKernels<dim, number> OperatorType>
  void
  compute_jacobian_matrix_representation(
    const OperatorType                                       &implicit_operator,
    std::variant<dealii::LinearAlgebra::distributed::Vector<number> *,
                 dealii::TrilinosWrappers::SparseMatrix *>    dst,
    const MatrixRepresentationType                            type,
    const dealii::LinearAlgebra::distributed::Vector<number> &current_solution,
    const dealii::MatrixFree<dim, number>                    &matrix_free,
    const unsigned int                                        dof_idx,
    const unsigned int                                        quad_idx)
  {
    FECellIntegrator<dim, n_components, number> phi(matrix_free, dof_idx, quad_idx);

    // ## Cell ## //
    std::function<void(FECellIntegrator<dim, n_components, number> &)> cell_operation =
      [&](FECellIntegrator<dim, n_components, number> &delta_phi) -> void {
      phi.reinit(delta_phi.get_cell_or_face_batch_id());
      phi.gather_evaluate(current_solution,
                          dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      delta_phi.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for (const unsigned int q_index : delta_phi.quadrature_point_indices())
        {
          implicit_operator.local_cell_jacobian_kernel(delta_phi,
                                                       phi,
                                                       q_index,
                                                       delta_phi.get_cell_or_face_batch_id());
        }

      delta_phi.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    };

    // ## Face ##//
    FEFaceIntegrator<dim, n_components, number> phi_m(matrix_free,
                                                      true /*is_interior_face*/,
                                                      dof_idx,
                                                      quad_idx);
    FEFaceIntegrator<dim, n_components, number> phi_p(matrix_free,
                                                      false /*is_interior_face*/,
                                                      dof_idx,
                                                      quad_idx);
    std::function<void(FEFaceIntegrator<dim, n_components, number> &,
                       FEFaceIntegrator<dim, n_components, number> &)>
      face_operation = [&](FEFaceIntegrator<dim, n_components, number> &delta_phi_m,
                           FEFaceIntegrator<dim, n_components, number> &delta_phi_p) -> void {
      phi_m.reinit(delta_phi_m.get_cell_or_face_batch_id());
      phi_p.reinit(delta_phi_p.get_cell_or_face_batch_id());
      phi_m.gather_evaluate(current_solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      phi_p.gather_evaluate(current_solution,
                            dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      delta_phi_p.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      delta_phi_m.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);


      for (const unsigned int q_index : delta_phi_m.quadrature_point_indices())
        {
          implicit_operator.local_face_jacobian_kernel(
            delta_phi_m, delta_phi_p, phi_m, phi_p, q_index);
        }
      delta_phi_m.integrate(dealii::EvaluationFlags::values);
      delta_phi_p.integrate(dealii::EvaluationFlags::values);
    };

    // ## Boundary face ##//
    FEFaceIntegrator<dim, n_components, number> phi_boundary(matrix_free,
                                                             true /*is_interior_face*/,
                                                             dof_idx,
                                                             quad_idx);
    std::function<void(FEFaceIntegrator<dim, n_components, number> &)> boundary_operation =
      [&](FEFaceIntegrator<dim, n_components, number> &delta_phi_boundary) -> void {
      phi_boundary.reinit(delta_phi_boundary.get_cell_or_face_batch_id());
      phi_boundary.gather_evaluate(current_solution,
                                   dealii::EvaluationFlags::values |
                                     dealii::EvaluationFlags::gradients);
      delta_phi_boundary.evaluate(dealii::EvaluationFlags::values |
                                  dealii::EvaluationFlags::gradients);


      for (const unsigned int q_index : delta_phi_boundary.quadrature_point_indices())
        {
          implicit_operator.local_boundary_face_jacobian_kernel(delta_phi_boundary,
                                                                phi_boundary,
                                                                q_index);
        }
      delta_phi_boundary.integrate(dealii::EvaluationFlags::values);
    };

    switch (type)
      {
          case MatrixRepresentationType::DiagonalMatrix: {
            dealii::MatrixFreeTools::compute_diagonal(
              matrix_free,
              *std::get<dealii::LinearAlgebra::distributed::Vector<number> *>(dst),
              cell_operation,
              face_operation,
              boundary_operation,
              dof_idx,
              quad_idx);
            break;
          }
          case MatrixRepresentationType::SystemMatrix: {
            dealii::MatrixFreeTools::compute_matrix(
              matrix_free,
              dealii::AffineConstraints<number>(),
              *std::get<dealii::TrilinosWrappers::SparseMatrix *>(dst),
              cell_operation,
              face_operation,
              boundary_operation,
              dof_idx,
              quad_idx);
            break;
          }
        default:
          DEAL_II_ASSERT_UNREACHABLE();
      }
  }
} // namespace MeltPoolDG