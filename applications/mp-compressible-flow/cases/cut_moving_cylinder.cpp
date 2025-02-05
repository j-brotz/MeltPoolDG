#include "cut_moving_cylinder.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions
  template class SimulationCutMovingCylinder<2>;
  template class SimulationCutMovingCylinder<3>;
} // namespace MeltPoolDG::Simulation::CompressibleFlow