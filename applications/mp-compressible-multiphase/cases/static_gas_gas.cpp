#include "static_gas_gas.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  // Explicit instantiations for the required dimensions
  template class SimulationStaticGasGas<1>;
  template class SimulationStaticGasGas<2>;
  template class SimulationStaticGasGas<3>;
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase