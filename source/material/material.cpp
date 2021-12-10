#include <meltpooldg/material/material.hpp>


namespace MeltPoolDG
{
  MaterialTypes
  determine_material_type(const bool do_two_phase,
                          const bool do_solidification,
                          const bool do_evaporation)
  {
    if (do_two_phase && do_solidification && do_evaporation)
      return MaterialTypes::gas_liquid_solid_consistent_with_evaporation;
    else if (do_two_phase && do_solidification)
      return MaterialTypes::gas_liquid_solid;
    else if (do_two_phase && do_evaporation)
      return MaterialTypes::gas_liquid_consistent_with_evaporation;
    else if (do_two_phase)
      return MaterialTypes::gas_liquid;
    else if (do_solidification)
      return MaterialTypes::liquid_solid;
    else
      return MaterialTypes::single_phase;
  }
} // namespace MeltPoolDG