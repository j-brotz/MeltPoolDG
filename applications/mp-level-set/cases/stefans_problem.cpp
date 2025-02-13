#include "stefans_problem.hpp"

namespace MeltPoolDG::Simulation::StefansProblem
{
  // Explicit instantiations for the required dimensions
  template class SimulationStefansProblem<1>;
  template class SimulationStefansProblem<2>;
  template class SimulationStefansProblem<3>;
} // namespace MeltPoolDG::Simulation::StefansProblem
