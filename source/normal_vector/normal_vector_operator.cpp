#include <meltpooldg/normal_vector/normal_vector_operator.hpp>

namespace MeltPoolDG::NormalVector
{
  template class NormalVectorOperator<1, double>;
  template class NormalVectorOperator<2, double>;
  template class NormalVectorOperator<3, double>;
} // namespace MeltPoolDG::NormalVector
