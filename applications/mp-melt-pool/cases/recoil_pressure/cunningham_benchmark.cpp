#include "cunningham_benchmark.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::CunninghamBenchmark
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationCunninghamBenchmark, "cunningham", 1, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationCunninghamBenchmark, "cunningham", 2, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationCunninghamBenchmark, "cunningham", 3, double);
} // namespace MeltPoolDG::Simulation::CunninghamBenchmark
