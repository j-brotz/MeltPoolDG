#include "cut_flow_over_cylinder.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions
  template class SimulationCutFlowOverCylinder<2>;
  template class SimulationCutFlowOverCylinder<3>;
} // namespace MeltPoolDG::Simulation::CompressibleFlow