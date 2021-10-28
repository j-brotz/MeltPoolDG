#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>

namespace MeltPoolDG::Reinitialization
{
  template <int dim>
  void
  ReinitializationOperationBase<dim>::initialize(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const Parameters<double> &                     data_in,
    const unsigned int                             reinit_dof_idx_in,
    const unsigned int                             reinit_quad_idx_in,
    const unsigned int                             ls_dof_idx_in,
    const unsigned int                             normal_dof_idx_in)
  {
    (void)scratch_data_in;
    (void)data_in;
    (void)reinit_dof_idx_in;
    (void)reinit_quad_idx_in;
    (void)ls_dof_idx_in;
    (void)normal_dof_idx_in;
    AssertThrow(false, ExcNotImplemented());
  }


  template class ReinitializationOperationBase<1>;
  template class ReinitializationOperationBase<2>;
  template class ReinitializationOperationBase<3>;

} // namespace MeltPoolDG::Reinitialization
