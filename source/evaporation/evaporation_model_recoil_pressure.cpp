#include <meltpooldg/evaporation/evaporation_model_recoil_pressure.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>

namespace MeltPoolDG::Evaporation
{
  template <int dim>
  EvaporationModelRecoilPressure<dim>::EvaporationModelRecoilPressure(
    const double boiling_temperature,
    const double pressure_constant,
    const double temperature_constant,
    const double sticking_constant,
    const double molar_mass,
    const double mass_flux_scale_factor)
    : boiling_temperature(boiling_temperature)
    , pressure_constant(pressure_constant)
    , temperature_constant(temperature_constant)
    , mass_flux_scale_factor(mass_flux_scale_factor)
    , sticking_constant(sticking_constant)
    , Cm(molar_mass / (2. * numbers::PI * PhysicalConstants::universal_gas_constant))
  {}

  template <int dim>
  double
  EvaporationModelRecoilPressure<dim>::local_compute_evaporative_mass_flux(const double T) const
  {
    return (T >= boiling_temperature) ?
             mass_flux_scale_factor * 0.82 * sticking_constant *
               MeltPool::RecoilPressureOperation<dim>::compute_recoil_pressure_coefficient(
                 T, pressure_constant, temperature_constant, boiling_temperature) *
               std::sqrt(Cm / T) :
             0.0;
  }

  template class EvaporationModelRecoilPressure<1>;
  template class EvaporationModelRecoilPressure<2>;
  template class EvaporationModelRecoilPressure<3>;

} // namespace MeltPoolDG::Evaporation
