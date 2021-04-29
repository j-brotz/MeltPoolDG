/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, UIBK/TUM, April 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/heat_transfer_operator.hpp>

#include <variant>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class HeatTransferPreconditioner
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const ScratchData<dim> &scratch_data;
    /**
     * select the relevant DoFHandlers
     */
    const unsigned int temp_dof_idx;

    DynamicSparsityPattern         dsp;
    TrilinosWrappers::SparseMatrix preconditioner_system_matrix;

  public:
    HeatTransferPreconditioner(const ScratchData<dim> &scratch_data_in,
                               const unsigned int      temp_dof_idx_in)
      : scratch_data(scratch_data_in)
      , temp_dof_idx(temp_dof_idx_in)
    {
      reinit();
    }

    void
    reinit()
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

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const
    {
      return preconditioner_system_matrix;
    }

    TrilinosWrappers::SparseMatrix &
    get_system_matrix()
    {
      return preconditioner_system_matrix;
    }

    DiagonalMatrix<VectorType>
    get_diagonal_preconditioner(const std::string &preconditioner_type,
                                const std::shared_ptr<HeatTransferOperator<dim>> &heat_operator)
    {
      if (preconditioner_type == "Diagonal")
        {
          using Preconditioner = DiagonalMatrix<VectorType>;

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
  };
} // namespace MeltPoolDG::Heat
