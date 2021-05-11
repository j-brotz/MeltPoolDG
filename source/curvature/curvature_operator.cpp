#include <meltpooldg/curvature/curvature_operator.hpp>

namespace MeltPoolDG::Curvature
{
  template class CurvatureOperator<1, double>;
  template class CurvatureOperator<2, double>;
  template class CurvatureOperator<3, double>;
} // namespace MeltPoolDG::Curvature
