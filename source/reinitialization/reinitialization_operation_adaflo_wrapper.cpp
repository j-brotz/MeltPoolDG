#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/reinitialization/reinitialization_operation_adaflo_wrapper.hpp>

namespace MeltPoolDG::Reinitialization
{
  template class ReinitializationOperationAdaflo<1>;
  template class ReinitializationOperationAdaflo<2>;
  template class ReinitializationOperationAdaflo<3>;
} // namespace MeltPoolDG::Reinitialization
#endif
