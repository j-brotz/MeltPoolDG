#include <deal.II/base/numbers.h>

#include <meltpooldg/interface/operator_base.hpp>
namespace MeltPoolDG
{
  template <int dim, typename number>
  void
  OperatorBase<dim, number>::initialize_matrix_based(const ScratchData<dim> &scratch_data)
  {
    AssertThrow(this->dof_idx < numbers::invalid_unsigned_int,
                ExcMessage("reset_dof_index() must be called."));
    const MPI_Comm mpi_communicator = scratch_data.get_mpi_comm(this->dof_idx);
    dsp.reinit(scratch_data.get_locally_owned_dofs(this->dof_idx),
               scratch_data.get_locally_owned_dofs(this->dof_idx),
               scratch_data.get_locally_relevant_dofs(this->dof_idx),
               mpi_communicator);

    DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(this->dof_idx),
                                    this->dsp,
                                    scratch_data.get_constraint(this->dof_idx),
                                    true,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));
    this->dsp.compress();

    this->system_matrix.reinit(dsp);
  }

  template class OperatorBase<1, double>;
  template class OperatorBase<2, double>;
  template class OperatorBase<3, double>;
} // namespace MeltPoolDG
