#include "static_liquid_gas.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  // Explicit instantiations for the required dimensions
  template class SimulationStaticLiquidGas<1>;
  template class SimulationStaticLiquidGas<2>;
  template class SimulationStaticLiquidGas<3>;
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase