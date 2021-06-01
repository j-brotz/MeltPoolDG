#include <deal.II/base/numbers.h>

#include <meltpooldg/evaporation/evaporation_model_hardt_wondra.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  EvaporationModelHardtWondra::EvaporationModelHardtWondra(const double evaporation_coefficient,
                                                           const double latent_heat_of_evaporation,
                                                           const double density_vapor,
                                                           const double molar_mass_vapor,
                                                           const double boiling_temperature)
    : evaporative_mass_transfer_coefficient(
        2. * evaporation_coefficient * latent_heat_of_evaporation * density_vapor /
        ((2. - evaporation_coefficient) *
         std::sqrt(2. * dealii::numbers::PI * PhysicalConstants::universal_gas_constant /
                   molar_mass_vapor) *
         std::pow(boiling_temperature, 1.5)))
    , boiling_temperature(boiling_temperature)
  {}

  EvaporationModelHardtWondra::EvaporationModelHardtWondra(
    const double evaporative_mass_transfer_coefficient,
    const double boiling_temperature)
    : evaporative_mass_transfer_coefficient(evaporative_mass_transfer_coefficient)
    , boiling_temperature(boiling_temperature)
  {}

  double
  EvaporationModelHardtWondra::local_compute_evaporative_mass_flux(const double T)
  {
    return (T >= boiling_temperature) ?
             evaporative_mass_transfer_coefficient * (T - boiling_temperature) :
             0.0;
  }

} // namespace MeltPoolDG::Evaporation
