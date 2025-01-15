#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <meltpooldg/phase_change/evaporation_model_hardt_wondra.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <typename number>
  EvaporationModelHardtWondra<number>::EvaporationModelHardtWondra(
    const number evaporation_coefficient,
    const number latent_heat_of_evaporation,
    const number density_vapor,
    const number molar_mass_vapor,
    const number boiling_temperature)
    : evaporative_mass_transfer_coefficient(
        2. * evaporation_coefficient * latent_heat_of_evaporation * density_vapor /
        ((2. - evaporation_coefficient) *
         std::sqrt(2. * dealii::numbers::PI * PhysicalConstants::universal_gas_constant /
                   molar_mass_vapor) *
         std::pow(boiling_temperature, 1.5)))
    , boiling_temperature(boiling_temperature)
  {
    AssertThrow(std::abs(2. - evaporation_coefficient) > 1e-12,
                ExcMessage("The evaporation coefficient must not be equal to two."));
    AssertThrow(molar_mass_vapor > 1e-12, ExcMessage("The molar mass must be larger than zero."));
    AssertThrow(
      std::abs(boiling_temperature) > 1e-12,
      ExcMessage(
        "The boiling temperature must not be zero to compute the evaporation mass transfer coefficient"));
  }

  template <typename number>
  EvaporationModelHardtWondra<number>::EvaporationModelHardtWondra(
    const number evaporative_mass_transfer_coefficient,
    const number boiling_temperature)
    : evaporative_mass_transfer_coefficient(evaporative_mass_transfer_coefficient)
    , boiling_temperature(boiling_temperature)
  {}


  template <typename number>
  number
  EvaporationModelHardtWondra<number>::local_compute_evaporative_mass_flux(const number T) const
  {
    return (T >= boiling_temperature) ?
             evaporative_mass_transfer_coefficient * (T - boiling_temperature) :
             0.0;
  }


  template class EvaporationModelHardtWondra<double>;
} // namespace MeltPoolDG::Evaporation
