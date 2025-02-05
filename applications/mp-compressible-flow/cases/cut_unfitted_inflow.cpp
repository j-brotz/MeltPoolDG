#include "cut_unfitted_inflow.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  // Explicit instantiations for the required dimensions
  template class SimulationCutUnfittedInflow<2>;
  template class SimulationCutUnfittedInflow<3>;
} // namespace MeltPoolDG::Simulation::CompressibleFlow