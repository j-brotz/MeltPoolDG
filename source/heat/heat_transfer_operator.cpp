#include <meltpooldg/heat/heat_transfer_operator.hpp>

namespace MeltPoolDG::Heat
{
  template class HeatTransferOperator<1>;
  template class HeatTransferOperator<2>;
  template class HeatTransferOperator<3>;
} // namespace MeltPoolDG::Heat
