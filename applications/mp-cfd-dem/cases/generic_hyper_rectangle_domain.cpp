#include "generic_hyper_rectangle_domain.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CfdDem
{
  // Explicit instantiations for the required dimensions and floating point number type
  MELTPOOLDG_REGISTER_CASE(CfdDemCase,
                           SimulationGenericHyperRectangleDomain,
                           "generic_hyper_rectangle_domain",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(CfdDemCase,
                           SimulationGenericHyperRectangleDomain,
                           "generic_hyper_rectangle_domain",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::CfdDem