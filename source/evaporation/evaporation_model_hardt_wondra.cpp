#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <meltpooldg/evaporation/evaporation_model_hardt_wondra.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

#include <cmath>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

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
  {
    AssertThrow(std::abs(2. - evaporation_coefficient) > 1e-12,
                ExcMessage("The evaporation coefficient must not be equal to two."));
    AssertThrow(molar_mass_vapor > 1e-12, ExcMessage("The molar mass must be larger than zero."));
    AssertThrow(
      std::abs(boiling_temperature) > 1e-12,
      ExcMessage(
        "The boiling temperature must not be zero to compute the evaporation mass transfer coefficient"));
  }

  EvaporationModelHardtWondra::EvaporationModelHardtWondra(
    const double evaporative_mass_transfer_coefficient,
    const double boiling_temperature)
    : evaporative_mass_transfer_coefficient(evaporative_mass_transfer_coefficient)
    , boiling_temperature(boiling_temperature)
  {}

  double
  EvaporationModelHardtWondra::local_compute_evaporative_mass_flux(const double T) const
  {
    return (T >= boiling_temperature) ?
             evaporative_mass_transfer_coefficient * (T - boiling_temperature) :
             0.0;
  }

} // namespace MeltPoolDG::Evaporation
