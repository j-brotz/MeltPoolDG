#include "film_boiling.hpp"

#include <meltpooldg/core/case_registration.hpp>

namespace MeltPoolDG::Simulation::FilmBoiling
{
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationFilmBoiling, "film_boiling", 1, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationFilmBoiling, "film_boiling", 2, double);
  MELTPOOLDG_REGISTER_CASE(MeltPoolCase, SimulationFilmBoiling, "film_boiling", 3, double);
} // namespace MeltPoolDG::Simulation::FilmBoiling
