#pragma once

#include <meltpooldg/phase_change/evaporative_heat_loss.hpp>
//
#include <deal.II/base/exceptions.h>

#include <meltpooldg/utilities/numbers.hpp>


namespace MeltPoolDG::Evaporation
{
  template <typename number>
  EvaporativeHeatLoss<number>::EvaporativeHeatLoss(const EvaporationData<number> &evapor_data,
                                                   const MaterialData<number>    &material_data,
                                                   const bool setup_internal_mass_flux_operator)
    : do_phenomenological_recoil_pressure(
        evapor_data.evaporative_cooling.consider_enthalpy_transport_vapor_mass_flux == "true")
    , latent_heat_of_evaporation(material_data.latent_heat_of_evaporation)
    , specific_heat_capacity(material_data.liquid.specific_heat_capacity)
    , specific_enthalpy_reference_temperature(material_data.specific_enthalpy_reference_temperature)
  {
    if (do_phenomenological_recoil_pressure)
      AssertThrow(!dealii::numbers::is_invalid(specific_enthalpy_reference_temperature),
                  dealii::ExcMessage(
                    "For the phenomenological recoil pressure model, the reference temperature "
                    "for computing the specific enthalpy must be specified. Abort..."));

    if (setup_internal_mass_flux_operator)
      {
        AssertThrow(evapor_data.evaporative_mass_flux_model ==
                      EvaporationModelType::recoil_pressure,
                    dealii::ExcNotImplemented());

        mass_flux_operator = std::make_unique<EvaporationModelRecoilPressure<number>>(
          evapor_data.recoil,
          material_data.boiling_temperature,
          material_data.molar_mass,
          material_data.latent_heat_of_evaporation);
      }
  }


  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeHeatLoss<number>::compute_evaporative_heat_loss(
    const ValueType                  &mass_flux,
    [[maybe_unused]] const ValueType &temperature) const
  {
    ValueType specific_enthalpy(0.0);
    if (do_phenomenological_recoil_pressure)
      specific_enthalpy = compute_phenomenological_specific_enthalpy(temperature);

    return -(latent_heat_of_evaporation + specific_enthalpy) * mass_flux;
  }


  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeHeatLoss<number>::compute_evaporative_heat_loss(const ValueType &temperature) const
  {
    Assert(mass_flux_operator,
           dealii::ExcMessage("To use this function, the class must be constructed with "
                              "setup_internal_mass_flux_operator = true."));

    return compute_evaporative_heat_loss(
      mass_flux_operator->local_compute_evaporative_mass_flux(temperature), temperature);
  }


  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeHeatLoss<number>::compute_evaporative_heat_loss_derivative_constant_mass_flux(
    [[maybe_unused]] const ValueType &mass_flux) const
  {
    if (do_phenomenological_recoil_pressure)
      return -specific_heat_capacity * mass_flux;
    else
      return 0.0;
  }


  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeHeatLoss<number>::
    compute_evaporative_heat_loss_derivative_with_temperature_dependent_mass_flux(
      [[maybe_unused]] const ValueType &temperature) const
  {
    Assert(mass_flux_operator,
           dealii::ExcMessage("To use this function, the class must be constructed with "
                              "setup_internal_mass_flux_operator = true."));

    const auto mass_flux_derivative =
      mass_flux_operator->local_compute_evaporative_mass_flux_derivative(temperature);

    if (do_phenomenological_recoil_pressure)
      return -specific_heat_capacity *
               mass_flux_operator->local_compute_evaporative_mass_flux(temperature) -
             (latent_heat_of_evaporation +
              compute_phenomenological_specific_enthalpy(temperature)) *
               mass_flux_derivative;
    else
      return -specific_heat_capacity * mass_flux_derivative;
  }


  template <typename number>
  template <typename ValueType>
  inline ValueType
  EvaporativeHeatLoss<number>::compute_phenomenological_specific_enthalpy(
    const ValueType &temperature) const
  {
    return specific_heat_capacity * (temperature - specific_enthalpy_reference_temperature);
  }

} // namespace MeltPoolDG::Evaporation