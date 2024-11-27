#include "advection_diffusion_DG.hpp"

namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
{
  // Explicit instantiations for the required dimensions
  template class SimulationAdvecDG<1>;
  template class SimulationAdvecDG<2>;
  template class SimulationAdvecDG<3>;
} // namespace MeltPoolDG::Simulation::AdvectionDiffusionDG
