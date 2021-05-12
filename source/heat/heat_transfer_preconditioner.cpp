#include <meltpooldg/heat/heat_transfer_preconditioner.hpp>

namespace MeltPoolDG::Heat
{
  template class HeatTransferPreconditioner<1>;
  template class HeatTransferPreconditioner<2>;
  template class HeatTransferPreconditioner<3>;
} // namespace MeltPoolDG::Heat
