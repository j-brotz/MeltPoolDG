/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, UIBK/TUM, April 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/heat_transfer_operator.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

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
                               unsigned int            temp_dof_idx_in);

    void
    reinit();

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const;

    TrilinosWrappers::SparseMatrix &
    get_system_matrix();

    DiagonalMatrix<VectorType>
    get_diagonal_preconditioner(const std::string &preconditioner_type,
                                const std::shared_ptr<HeatTransferOperator<dim>> &heat_operator);
  };
} // namespace MeltPoolDG::Heat
