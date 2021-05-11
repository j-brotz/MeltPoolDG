#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/flow/adaflo_wrapper.hpp>

namespace MeltPoolDG::Flow
{
  template class AdafloWrapper<1>;
  template class AdafloWrapper<2>;
  template class AdafloWrapper<3>;
} // namespace MeltPoolDG::Flow
#endif
