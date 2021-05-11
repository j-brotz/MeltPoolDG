#include <meltpooldg/reinitialization/olsson_operator.hpp>

namespace MeltPoolDG::Reinitialization
{
  template class OlssonOperator<1, double>;
  template class OlssonOperator<2, double>;
  template class OlssonOperator<3, double>;
} // namespace MeltPoolDG::Reinitialization
