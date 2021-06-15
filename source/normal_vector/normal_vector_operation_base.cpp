#include <meltpooldg/normal_vector/normal_vector_operation_base.hpp>
#include <meltpooldg/utilities/journal.hpp>

namespace MeltPoolDG::NormalVector
{
  template <int dim>
  void
  NormalVectorOperationBase<dim>::initialize(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &                     data_in,
    const unsigned int                             normal_dof_idx_in,
    const unsigned int                             normal_quad_idx_in,
    const unsigned int                             ls_dof_idx_in)
  {
    (void)scratch_data_in;
    (void)data_in;
    (void)normal_dof_idx_in;
    (void)normal_quad_idx_in;
    (void)ls_dof_idx_in;
    AssertThrow(false, ExcNotImplemented());
  }


  template class NormalVectorOperationBase<1>;
  template class NormalVectorOperationBase<2>;
  template class NormalVectorOperationBase<3>;
} // namespace MeltPoolDG::NormalVector
