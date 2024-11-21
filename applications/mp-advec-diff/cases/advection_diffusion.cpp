#include "advection_diffusion.hpp"

namespace MeltPoolDG::Simulation::AdvectionDiffusion
{
  // Explicit instantiations for the required dimensions
  template class SimulationAdvec<1>;
  template class SimulationAdvec<2>;
  template class SimulationAdvec<3>;
} // namespace MeltPoolDG::Simulation::AdvectionDiffusion
