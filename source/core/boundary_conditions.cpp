#include <meltpooldg/core/boundary_conditions.hpp>

#include <algorithm>

namespace MeltPoolDG
{
  template struct BoundaryConditionManager<1, double>;
  template struct BoundaryConditionManager<2, double>;
  template struct BoundaryConditionManager<3, double>;
} // namespace MeltPoolDG
