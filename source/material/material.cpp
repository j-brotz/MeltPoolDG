#include <deal.II/base/exceptions.h>

#include <meltpooldg/material/material.hpp>

namespace MeltPoolDG
{
  MaterialTypes
  determine_material_type(const bool do_two_phase,
                          const bool do_solidification,
                          const bool do_evaporation)
  {
    if (do_two_phase)
      {
        if (do_solidification)
          {
            if (do_evaporation)
              return MaterialTypes::gas_liquid_solid_consistent_with_evaporation;
            else
              return MaterialTypes::gas_liquid_solid;
          }
        else // do_solidification == false
          {
            if (do_evaporation)
              return MaterialTypes::gas_liquid_consistent_with_evaporation;
            else
              return MaterialTypes::gas_liquid;
          }
      }
    else // do_two_phase == false
      {
        Assert(do_evaporation == false,
               dealii::ExcMessage(
                 "In the case that no two phase flow is enabled, the material cannot be determined "
                 "consistent with evaporation! Abort..."));
        if (do_solidification)
          return MaterialTypes::liquid_solid;
        else
          return MaterialTypes::single_phase;
      }
  }



  template <typename number>
  const MaterialData<number> &
  Material<number>::get_data() const
  {
    return data;
  }



  template class Material<double>;
} // namespace MeltPoolDG
