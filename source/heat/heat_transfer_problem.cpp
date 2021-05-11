#include <meltpooldg/heat/heat_transfer_problem.hpp>

namespace MeltPoolDG::Heat
{
  template class HeatTransferProblem<1>;
  template class HeatTransferProblem<2>;
  template class HeatTransferProblem<3>;
} // namespace MeltPoolDG::Heat
