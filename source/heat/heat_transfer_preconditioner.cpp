#include <meltpooldg/heat/heat_transfer_preconditioner.hpp>
//

namespace MeltPoolDG::Heat
{
  template <int dim>
  HeatTransferPreconditioner<dim>::HeatTransferPreconditioner(
    const ScratchData<dim> &scratch_data_in,
    const unsigned int      temp_dof_idx_in)
    : scratch_data(scratch_data_in)
    , temp_dof_idx(temp_dof_idx_in)
  {
    reinit();
  }

  template <int dim>
  void
  HeatTransferPreconditioner<dim>::reinit()
  {
    const MPI_Comm mpi_communicator = scratch_data.get_mpi_comm(temp_dof_idx);

    dsp.reinit(scratch_data.get_dof_handler(temp_dof_idx).n_dofs(),
               scratch_data.get_dof_handler(temp_dof_idx).n_dofs());
    DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(temp_dof_idx),
                                    dsp,
                                    scratch_data.get_constraint(temp_dof_idx));

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               scratch_data.get_locally_owned_dofs(temp_dof_idx),
                                               scratch_data.get_mpi_comm(),
                                               scratch_data.get_locally_relevant_dofs(
                                                 temp_dof_idx));

    preconditioner_system_matrix.reinit(scratch_data.get_locally_owned_dofs(temp_dof_idx),
                                        scratch_data.get_locally_owned_dofs(temp_dof_idx),
                                        dsp,
                                        mpi_communicator);
  }

  template <int dim>
  const TrilinosWrappers::SparseMatrix &
  HeatTransferPreconditioner<dim>::get_system_matrix() const
  {
    return preconditioner_system_matrix;
  }

  template <int dim>
  TrilinosWrappers::SparseMatrix &
  HeatTransferPreconditioner<dim>::get_system_matrix()
  {
    return preconditioner_system_matrix;
  }

  template <int dim>
  DiagonalMatrix<VectorType>
  HeatTransferPreconditioner<dim>::get_diagonal_preconditioner(
    const std::string &                               preconditioner_type,
    const std::shared_ptr<HeatTransferOperator<dim>> &heat_operator)
  {
    if (preconditioner_type == "Diagonal")
      {
        heat_operator->compute_system_matrix(preconditioner_system_matrix);

        VectorType diag;
        scratch_data.initialize_dof_vector(diag, temp_dof_idx);

        for (const auto i : diag.locally_owned_elements())
          diag[i] = preconditioner_system_matrix(i, i);
        for (auto &i : diag)
          i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;

        diag.update_ghost_values();

        return DiagonalMatrix<VectorType>(diag);
      }
    else if (preconditioner_type == "DiagonalReduced")
      {
        using Preconditioner = DiagonalMatrix<VectorType>;

        VectorType diag;
        heat_operator->compute_inverse_diagonal(diag);

        return DiagonalMatrix<VectorType>(diag);
      }
    else
      AssertThrow(false, ExcNotImplemented());
  }

  template class HeatTransferPreconditioner<1>;
  template class HeatTransferPreconditioner<2>;
  template class HeatTransferPreconditioner<3>;
} // namespace MeltPoolDG::Heat
