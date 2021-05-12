#include <meltpooldg/interface/periodic_boundary_conditions.hpp>

namespace MeltPoolDG
{
  template struct PeriodicBoundaryConditions<1>;
  template struct PeriodicBoundaryConditions<2>;
  template struct PeriodicBoundaryConditions<3>;
} // namespace MeltPoolDG
