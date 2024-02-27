#include <meltpooldg/evaporation/evaporation_model_saturated_vapor_pressure.hpp>
#include <meltpooldg/evaporation/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

namespace MeltPoolDG::Evaporation
{
  EvaporationModelSaturatedVaporPressure::EvaporationModelSaturatedVaporPressure(
    const RecoilPressureData<double> &recoil_data,
    const double                      boiling_temperature,
    const double                      sticking_constant,
    const double                      molar_mass,
    const double                      latent_heat_evaporation,
    const double                      mass_flux_scale_factor)
    : recoil_data(recoil_data)
    , boiling_temperature(boiling_temperature)
    , sticking_constant(sticking_constant)
    , molar_mass(molar_mass)
    , latent_heat_evaporation(latent_heat_evaporation)
    , mass_flux_scale_factor(mass_flux_scale_factor)
    , Cm(molar_mass / (2. * numbers::PI * PhysicalConstants::universal_gas_constant))
  {}

  double
  EvaporationModelSaturatedVaporPressure::local_compute_evaporative_mass_flux(const double T) const
  {
    return mass_flux_scale_factor * 0.82 * sticking_constant *
           compute_saturated_gas_pressure(T,
                                          boiling_temperature,
                                          molar_mass,
                                          latent_heat_evaporation,
                                          recoil_data.ambient_gas_pressure,
                                          recoil_data.temperature_constant) *
           std::sqrt(Cm / T);
  }
} // namespace MeltPoolDG::Evaporation
