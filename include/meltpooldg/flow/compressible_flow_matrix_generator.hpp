/**
 * @brief Collection of functions that create a matrix representaiton (diagonal, sparse system
 * matrix) for a matrix free object.
 */

#pragma once

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * An enum of possible matrix representation types that can be computed from a matrix free object.
   */
  enum MatrixRepresentationType
  {
    DiagonalMatrix,
    SystemMatrix
  };

  /**
   * A concept setting the requirements on a class for which the matrix representation of the
   * jacobian shall be computed with the function compute_jacobian_matrix_representation().
   */
  template <typename OperatorType, int dim, typename number>
  concept HasJacobianQuadraturePointKernels =
    requires(OperatorType                                                 op,
             dealii::FEEvaluation<dim, -1, 0, dim + 2, number>           &fe_evaluator,
             const dealii::FEEvaluation<dim, -1, 0, dim + 2, number>     &const_fe_evaluator,
             dealii::FEFaceEvaluation<dim, -1, 0, dim + 2, number>       &fe_face_evaluator,
             const dealii::FEFaceEvaluation<dim, -1, 0, dim + 2, number> &const_fe_face_evaluator,
             unsigned int                                                 q_index) {
      /**
       * Compute the operations for computing the jacobian on the quadrature points for the cell
       * loop.
       */
      op.local_cell_jacobian_kernel(fe_evaluator, const_fe_evaluator, q_index);

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
    dealii::FECellIntegrator<dim, n_components, number> phi(matrix_free, dof_idx, quad_idx);

    // ## Cell ## //
    std::function<void(dealii::FECellIntegrator<dim, n_components, number> &)> cell_operation =
      [&](dealii::FECellIntegrator<dim, n_components, number> &delta_phi) -> void {
      phi.reinit(delta_phi.get_cell_or_face_batch_id());
      phi.gather_evaluate(current_solution,
                          dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
      delta_phi.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      for (const unsigned int q_index : delta_phi.quadrature_point_indices())
        {
          implicit_operator.local_cell_jacobian_kernel(delta_phi, phi, q_index);
        }

      delta_phi.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    };

    // ## Face ##//
    dealii::FEFaceIntegrator<dim, n_components, number> phi_m(matrix_free,
                                                              true /*is_interior_face*/,
                                                              dof_idx,
                                                              quad_idx);
    dealii::FEFaceIntegrator<dim, n_components, number> phi_p(matrix_free,
                                                              false /*is_interior_face*/,
                                                              dof_idx,
                                                              quad_idx);
    std::function<void(dealii::FEFaceIntegrator<dim, n_components, number> &,
                       dealii::FEFaceIntegrator<dim, n_components, number> &)>
      face_operation =
        [&](dealii::FEFaceIntegrator<dim, n_components, number> &delta_phi_m,
            dealii::FEFaceIntegrator<dim, n_components, number> &delta_phi_p) -> void {
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
    dealii::FEFaceIntegrator<dim, n_components, number> phi_boundary(matrix_free,
                                                                     true /*is_interior_face*/,
                                                                     dof_idx,
                                                                     quad_idx);
    std::function<void(dealii::FEFaceIntegrator<dim, n_components, number> &)> boundary_operation =
      [&](dealii::FEFaceIntegrator<dim, n_components, number> &delta_phi_boundary) -> void {
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
} // namespace MeltPoolDG::Flow
