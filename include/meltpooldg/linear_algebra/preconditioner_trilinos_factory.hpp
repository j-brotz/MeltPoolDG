#pragma once
// for distributed vectors/matrices
#include <deal.II/lac/generic_linear_algebra.h>
// #include <deal.II/lac/la_parallel_block_vector.h>
// #include <deal.II/lac/la_parallel_vector.h>
// #include <deal.II/lac/trilinos_precondition.h>
// #include <deal.II/lac/trilinos_sparse_matrix.h>
//

#include <deal.II/base/exceptions.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>

#include <memory>

using namespace dealii;

namespace MeltPoolDG::Preconditioner
{
  inline std::shared_ptr<TrilinosWrappers::PreconditionBase>
  get_preconditioner_trilinos(const TrilinosWrappers::SparseMatrix &matrix,
                              PreconditionerType preconditioner_type = PreconditionerType::Identity)
  {
    switch (preconditioner_type)
      {
          case PreconditionerType::Identity: {
            return std::make_shared<TrilinosWrappers::PreconditionIdentity>();
          }
          case PreconditionerType::AMG: {
            auto preconditioner = std::make_shared<TrilinosWrappers::PreconditionAMG>();
            TrilinosWrappers::PreconditionAMG::AdditionalData data;
            preconditioner->initialize(matrix, data);
            return preconditioner;
            break;
          }
          case PreconditionerType::ILU: {
            auto preconditioner = std::make_shared<TrilinosWrappers::PreconditionILU>();
            TrilinosWrappers::PreconditionILU::AdditionalData data;
            preconditioner->initialize(matrix, data);
            return preconditioner;
            break;
          }
          default: {
            AssertThrow(false, ExcMessage("The requested preconditioner type is not implemented."));
            return nullptr;
            break;
          }
      }
  }
} // namespace MeltPoolDG::Preconditioner
