#include <meltpooldg/curvature/curvature_operator.hpp>
#include <meltpooldg/linear_algebra/preconditioner_matrixfree_generic.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_factory.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Preconditioner
{
  template <int dim, typename OperatorType>
  PreconditionerMatrixFreeGeneric<dim, OperatorType>::PreconditionerMatrixFreeGeneric(
    const ScratchData<dim>   &scratch_data_in,
    const unsigned int        dof_idx,
    const PreconditionerType &preconditioner_type_in,
    const OperatorType       &operator_base_in)
    : scratch_data(scratch_data_in)
    , dof_idx(dof_idx)
    , preconditioner_type(preconditioner_type_in)
    , operator_base(operator_base_in)
  {
    AssertThrow(preconditioner_type == PreconditionerType::Diagonal ||
                  preconditioner_type == PreconditionerType::ILU ||
                  preconditioner_type == PreconditionerType::AMG ||
                  preconditioner_type == PreconditionerType::Identity,
                ExcMessage("The supported preconditioner types are Diagonal|ILU|AMG|Identity."));
  }

  template <int dim, typename OperatorType>
  void
  PreconditionerMatrixFreeGeneric<dim, OperatorType>::reinit()
  {
    if (preconditioner_type == PreconditionerType::AMG ||
        preconditioner_type == PreconditionerType::ILU)
      {
        const MPI_Comm mpi_communicator = scratch_data.get_mpi_comm();

        dsp.reinit(scratch_data.get_dof_handler(dof_idx).n_dofs(),
                   scratch_data.get_dof_handler(dof_idx).n_dofs());
        DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(dof_idx),
                                        dsp,
                                        scratch_data.get_constraint(dof_idx));

        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   scratch_data.get_locally_owned_dofs(dof_idx),
                                                   scratch_data.get_mpi_comm(),
                                                   scratch_data.get_locally_relevant_dofs(dof_idx));

        preconditioner_system_matrix.reinit(scratch_data.get_locally_owned_dofs(dof_idx),
                                            scratch_data.get_locally_owned_dofs(dof_idx),
                                            dsp,
                                            mpi_communicator);
      }
  }

  template <int dim, typename OperatorType>
  std::shared_ptr<TrilinosWrappers::PreconditionBase>
  PreconditionerMatrixFreeGeneric<dim, OperatorType>::compute_trilinos_preconditioner()
  {
    if (preconditioner_type == PreconditionerType::AMG ||
        preconditioner_type == PreconditionerType::ILU)
      operator_base.compute_system_matrix_from_matrixfree(preconditioner_system_matrix);

    return Preconditioner::get_preconditioner_trilinos(preconditioner_system_matrix,
                                                       preconditioner_type);
  }

  template <int dim, typename OperatorType>
  const TrilinosWrappers::SparseMatrix &
  PreconditionerMatrixFreeGeneric<dim, OperatorType>::get_system_matrix() const
  {
    return preconditioner_system_matrix;
  }

  template <int dim, typename OperatorType>
  TrilinosWrappers::SparseMatrix &
  PreconditionerMatrixFreeGeneric<dim, OperatorType>::get_system_matrix()
  {
    return preconditioner_system_matrix;
  }

  template <int dim, typename OperatorType>
  std::shared_ptr<DiagonalMatrix<VectorType>>
  PreconditionerMatrixFreeGeneric<dim, OperatorType>::compute_diagonal_preconditioner()
  {
    VectorType diag;

    operator_base.compute_inverse_diagonal_from_matrixfree(diag);

    return std::make_shared<DiagonalMatrix<VectorType>>(diag);
  }

  template <int dim, typename OperatorType>
  std::shared_ptr<DiagonalMatrix<BlockVectorType>>
  PreconditionerMatrixFreeGeneric<dim, OperatorType>::compute_block_diagonal_preconditioner()
  {
    BlockVectorType diag;

    operator_base.compute_inverse_diagonal_from_matrixfree(diag);

    return std::make_shared<DiagonalMatrix<BlockVectorType>>(diag);
  }

  template class PreconditionerMatrixFreeGeneric<1, OperatorBase<1, double>>;
  template class PreconditionerMatrixFreeGeneric<2, OperatorBase<2, double>>;
  template class PreconditionerMatrixFreeGeneric<3, OperatorBase<3, double>>;

} // namespace MeltPoolDG::Preconditioner
