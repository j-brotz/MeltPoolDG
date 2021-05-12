#ifdef MELT_POOL_DG_WITH_ADAFLO
#  include <meltpooldg/curvature/curvature_operation_adaflo_wrapper.hpp>

namespace MeltPoolDG::Curvature
{
  template class CurvatureOperationAdaflo<1>;
  template class CurvatureOperationAdaflo<2>;
  template class CurvatureOperationAdaflo<3>;
} // namespace MeltPoolDG::Curvature
#endif
