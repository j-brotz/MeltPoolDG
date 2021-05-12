#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>

namespace MeltPoolDG::NormalVector
{
  template class NormalVectorOperationAdaflo<1>;
  template class NormalVectorOperationAdaflo<2>;
  template class NormalVectorOperationAdaflo<3>;
} // namespace MeltPoolDG::NormalVector
#endif
