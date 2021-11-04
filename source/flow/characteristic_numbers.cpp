#include <meltpooldg/flow/characteristic_numbers.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  CharacteristicNumbers<number>::CharacteristicNumbers(
    const MaterialParameterValues<number> &material_in)
    : material(material_in)
  {}

  template <typename number>
  number
  CharacteristicNumbers<number>::Reynolds(const number &characteristic_velocity,
                                          const number &characteristic_length)
  {
    AssertThrow(material.viscosity >= std::numeric_limits<number>::epsilon(),
                ExcMessage("The dynamic viscosity must be >0. Abort ..."));
    return material.density * characteristic_length * characteristic_velocity / material.viscosity;
  }

  template <typename number>
  number
  CharacteristicNumbers<number>::Mach(const number &characteristic_velocity,
                                      const number &characteristic_length)
  {
    AssertThrow(material.conductivity >= std::numeric_limits<number>::epsilon(),
                ExcMessage("The conductivity must be >0. Abort ..."));
    return material.density * material.capacity * characteristic_length * characteristic_velocity /
           material.conductivity;
  }

  template <typename number>
  number
  CharacteristicNumbers<number>::capillary(const number &characteristic_velocity,
                                           const number &surface_tension_coefficient)
  {
    AssertThrow(surface_tension_coefficient >= std::numeric_limits<number>::epsilon(),
                ExcMessage("The surface tension coefficient must be >0. Abort ..."));
    return material.viscosity * characteristic_velocity / surface_tension_coefficient;
  }

  template class CharacteristicNumbers<double>;
} // namespace MeltPoolDG::Flow
