/**
 * @brief Factory function for the preconditioner.
 */

#pragma once

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/preconditioner_jacobi.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_wrapper.hpp>

namespace MeltPoolDG
{
  /**
   * A preconditoner factory function. This function creates a preconditioner based on the passed
   * preconditioner type and the flag indicating whether the preconditioner is applied in a
   * matrix-free or a matrix-based context.
   *
   * @param preconditioner_type Type of the desired preconditioner.
   * @param operator_in Operator that supports the required functions for the computations inside
   * the preconditioner (see specific preconditioner classes for additional information).
   * @param do_matrix_free A flag indicating if the operator shall be used in a matrix-free or
   * matrix-based way.
   *
   * @return Preconditioner object using the passed preconditioner type.
   * @throws Exception if the preconditioner type is not supported.
   */
  template <int dim, typename OperatorType, typename VectorType>
  Preconditioner<dim, VectorType>
  make_preconditioner(const PreconditionerType &preconditioner_type,
                      const OperatorType       *operator_in,
                      const bool                do_matrix_free = true)
  {
    switch (preconditioner_type)
      {
          case PreconditionerType::Identity: {
            return Preconditioner<dim, VectorType>(IdentityPreconditioner<dim, VectorType>());
          }
          case PreconditionerType::AMG: {
            return Preconditioner<dim, VectorType>(
              DealiiPreconditionerWrapper<dim,
                                          VectorType,
                                          dealii::TrilinosWrappers::PreconditionAMG,
                                          OperatorType>(operator_in, do_matrix_free));
          }
          case PreconditionerType::ILU: {
            return Preconditioner<dim, VectorType>(
              DealiiPreconditionerWrapper<dim,
                                          VectorType,
                                          dealii::TrilinosWrappers::PreconditionILU,
                                          OperatorType>(operator_in, do_matrix_free));
          }
          case PreconditionerType::Diagonal: {
            if constexpr (JacobiPreconditionerOperatorType<OperatorType, VectorType>)
              {
                if (do_matrix_free)
                  return Preconditioner<dim, VectorType>(
                    JacobiPreconditioner<dim, VectorType, OperatorType>(*operator_in));
              }
            return Preconditioner<dim, VectorType>(
              DealiiPreconditionerWrapper<dim,
                                          VectorType,
                                          dealii::TrilinosWrappers::PreconditionJacobi,
                                          OperatorType>(operator_in, false));
          }
          default: {
            AssertThrow(false,
                        dealii::ExcMessage("The requested preconditioner type is not"
                                           " implemented."));
          }
      }
  }
} // namespace MeltPoolDG