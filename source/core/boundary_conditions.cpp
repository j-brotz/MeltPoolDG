#include <meltpooldg/core/boundary_conditions.hpp>

#include <algorithm>

namespace MeltPoolDG
{
  template struct BoundaryConditionManager<1>;
  template struct BoundaryConditionManager<2>;
  template struct BoundaryConditionManager<3>;
} // namespace MeltPoolDG
