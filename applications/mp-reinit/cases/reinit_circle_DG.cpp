#include "reinit_circle_DG.hpp"

namespace MeltPoolDG::Simulation::ReinitCircleDG
{
  // Explicit instantiations for the required dimensions
  template class SimulationReinitDG<1>;
  template class SimulationReinitDG<2>;
  template class SimulationReinitDG<3>;
} // namespace MeltPoolDG::Simulation::ReinitCircleDG
