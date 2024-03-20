#include <meltpooldg/evaporation/evaporation_model_recoil_pressure.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

namespace MeltPoolDG::Evaporation
{
  EvaporationModelRecoilPressure::EvaporationModelRecoilPressure(
    const RecoilPressureData<double> &recoil_data,
    const double                      boiling_temperature,
    const double                      molar_mass,
    const double                      latent_heat_evaporation)
    : recoil_model(recoil_data, boiling_temperature, molar_mass, latent_heat_evaporation)
    , sticking_constant(recoil_data.sticking_constant)
    , Cm(molar_mass / (2. * numbers::PI * PhysicalConstants::universal_gas_constant))
  {}

  double
  EvaporationModelRecoilPressure::local_compute_evaporative_mass_flux(const double T) const
  {
    return 0.82 * sticking_constant * recoil_model.compute_recoil_pressure_coefficient(T) *
           std::sqrt(Cm / T); //@todo: replace recoil pressure by saturated vapor pressure
  }
} // namespace MeltPoolDG::Evaporation
