#include "zalesak_disk.hpp"

namespace MeltPoolDG::Simulation::ZalesakDisk
{
  // Explicit instantiations for the required dimensions
  template class SimulationZalesakDisk<1>;
  template class SimulationZalesakDisk<2>;
  template class SimulationZalesakDisk<3>;
} // namespace MeltPoolDG::Simulation::ZalesakDisk
