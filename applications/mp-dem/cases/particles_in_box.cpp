#include "particles_in_box.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::Dem
{
  // Explicit instantiations for the required dimensions and floating point number type
  MELTPOOLDG_REGISTER_CASE(DemCase, ParticlesInBox, "particles_in_box", 2, double);
  MELTPOOLDG_REGISTER_CASE(DemCase, ParticlesInBox, "particles_in_box", 3, double);
} // namespace MeltPoolDG::Simulation::Dem