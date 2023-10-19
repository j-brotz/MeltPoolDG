#include <meltpooldg/heat/heat_transfer_preconditioner_matrixfree.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_factory.hpp>

namespace MeltPoolDG::Heat
{
  template <int dim>
  HeatTransferPreconditionerMatrixFree<dim>::HeatTransferPreconditionerMatrixFree(
    const ScratchData<dim>   &scratch_data_in,
    const unsigned int        temp_dof_idx_in,
    const PreconditionerType &preconditioner_type_in,
    const OperatorType       &heat_operator_in)
    : scratch_data(scratch_data_in)
    , temp_dof_idx(temp_dof_idx_in)
    , preconditioner_type(preconditioner_type_in)
    , heat_operator(heat_operator_in)
  {
    if (preconditioner_type == PreconditionerType::AMG ||
        preconditioner_type == PreconditionerType::AMGReduced)
      preconditioner_base_name = PreconditionerType::AMG;
    else if (preconditioner_type == PreconditionerType::ILU ||
             preconditioner_type == PreconditionerType::ILUReduced)
      preconditioner_base_name = PreconditionerType::ILU;
    else
      preconditioner_base_name = preconditioner_type;

    AssertThrow(preconditioner_type == PreconditionerType::Identity ||
                  preconditioner_type == PreconditionerType::AMG ||
                  preconditioner_type == PreconditionerType::AMGReduced ||
                  preconditioner_type == PreconditionerType::Diagonal ||
                  preconditioner_type == PreconditionerType::DiagonalReduced ||
                  preconditioner_type == PreconditionerType::ILU ||
                  preconditioner_type == PreconditionerType::ILUReduced,
                ExcMessage("The supported preconditioner types are Identity|Diagonal"
                           "|DiagonalReduced|ILU|ILUReduced|AMG|AMGReduced."));

    reinit();
  }

  template <int dim>
  void
  HeatTransferPreconditionerMatrixFree<dim>::reinit()
  {
    if (preconditioner_type != PreconditionerType::DiagonalReduced &&
        preconditioner_type != PreconditionerType::Identity)
      {
        const MPI_Comm mpi_communicator = scratch_data.get_mpi_comm(temp_dof_idx);

        dsp.reinit(scratch_data.get_dof_handler(temp_dof_idx).n_dofs(),
                   scratch_data.get_dof_handler(temp_dof_idx).n_dofs());
        DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(temp_dof_idx),
                                        dsp,
                                        scratch_data.get_constraint(temp_dof_idx));

        SparsityTools::distribute_sparsity_pattern(
          dsp,
          scratch_data.get_locally_owned_dofs(temp_dof_idx),
          scratch_data.get_mpi_comm(),
          scratch_data.get_locally_relevant_dofs(temp_dof_idx));

        preconditioner_system_matrix.reinit(scratch_data.get_locally_owned_dofs(temp_dof_idx),
                                            scratch_data.get_locally_owned_dofs(temp_dof_idx),
                                            dsp,
                                            mpi_communicator);
      }
  }

  template <int dim>
  const TrilinosWrappers::SparseMatrix &
  HeatTransferPreconditionerMatrixFree<dim>::get_system_matrix() const
  {
    return preconditioner_system_matrix;
  }

  template <int dim>
  TrilinosWrappers::SparseMatrix &
  HeatTransferPreconditionerMatrixFree<dim>::get_system_matrix()
  {
    return preconditioner_system_matrix;
  }

  template <int dim>
  std::shared_ptr<DiagonalMatrix<VectorType>>
  HeatTransferPreconditionerMatrixFree<dim>::compute_diagonal_preconditioner()
  {
    if (preconditioner_type == PreconditionerType::Diagonal)
      {
        heat_operator->compute_system_matrix_from_matrixfree(preconditioner_system_matrix);

        VectorType diag;
        scratch_data.initialize_dof_vector(diag, temp_dof_idx);

        for (const auto i : diag.locally_owned_elements())
          diag[i] = preconditioner_system_matrix(i, i);
        for (auto &i : diag)
          i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;

        diag.update_ghost_values();

        return std::make_shared<DiagonalMatrix<VectorType>>(diag);
      }
    else if (preconditioner_type == PreconditionerType::DiagonalReduced)
      {
        VectorType diag;
        heat_operator->compute_inverse_diagonal_from_matrixfree(diag);

        return std::make_shared<DiagonalMatrix<VectorType>>(diag);
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template <int dim>
  std::shared_ptr<TrilinosWrappers::PreconditionBase>
  HeatTransferPreconditionerMatrixFree<dim>::compute_trilinos_preconditioner()
  {
    if (preconditioner_type != PreconditionerType::Identity)
      {
        if (preconditioner_type == preconditioner_base_name)
          heat_operator->compute_system_matrix_from_matrixfree(preconditioner_system_matrix);
        else
          heat_operator->compute_system_matrix_from_matrixfree_reduced(
            preconditioner_system_matrix);
      }
    return Preconditioner::get_preconditioner_trilinos(preconditioner_system_matrix,
                                                       preconditioner_base_name);
  }

  template class HeatTransferPreconditionerMatrixFree<1>;
  template class HeatTransferPreconditionerMatrixFree<2>;
  template class HeatTransferPreconditionerMatrixFree<3>;
} // namespace MeltPoolDG::Heat
