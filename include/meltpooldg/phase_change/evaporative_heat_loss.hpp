#pragma once

#include <deal.II/base/exceptions.h>

#include <meltpooldg/utilities/material_data.hpp>
#include <meltpooldg/utilities/numbers.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * Compute the heat sink due to evaporation
   *             .
   *    q_s =  - m · ( h_v + h(T))
   *
   * with the latent heat of evaporation h_v, the specific enthalpy
   *
   *          T
   *         /
   *        |
   * h(T) = | c_p  .  (1)
   *        |
   *       /
   *     T_ref
   *
   * where T_ref denotes an artificial reference temperature for
   * the specific enthalpy.
   *
   * h_v + h(T) is the total enthalpy leaving the system with the
   * evaporative mass flux. h(T) can be typically found in tables.
   * However, we assume for the computation of the specific
   * enthalpy a linear temperature-dependence
   *
   *    h(T) = h_ref + c_p * T .  (2)
   *
   * see also
   * Meier, Christoph, et al. "A novel smoothed particle hydrodynamics
   * formulation for thermo-capillary phase change problems with focus
   * on metal additive manufacturing melt pool modeling." CMAME 381
   * (2021).
   *
   * By setting (1) equal to (2) and assuming the heat capacity to
   * be temperature-independent, we obtain
   *
   *    h_ref = - c_p * T_ref
   *
   * Inserting into (2) yields
   *
   *    h(T) = c_p * (T - T_ref).
   *
   *
   * @note For the computation of h(T), it is assumed that the
   *       specific heat capacity c_p corresponds to the value
   *       for the liquid and solid phase.
   *
   * @note Instead of T_ref we could have also introduced directly
   *       h_ref as an input parameter.
   */
  template <typename number>
  class EvaporativeHeatLoss
  {
  public:
    EvaporativeHeatLoss(const bool                  do_phenomenological_recoil_pressure_in,
                        const MaterialData<number> &material_data)
      : do_phenomenological_recoil_pressure(do_phenomenological_recoil_pressure_in)
      , latent_heat_of_evaporation(material_data.latent_heat_of_evaporation)
      , specific_heat_capacity(material_data.liquid.specific_heat_capacity)
      , specific_enthalpy_reference_temperature(
          material_data.specific_enthalpy_reference_temperature)
    {
      if (do_phenomenological_recoil_pressure)
        AssertThrow(!dealii::numbers::is_invalid(specific_enthalpy_reference_temperature),
                    dealii::ExcMessage(
                      "For the phenomenological recoil pressure model, the reference temperature "
                      "for computing the specific enthalpy must be specified. Abort..."));
    }


    template <typename ValueType>
    inline ValueType
    compute_evaporative_heat_loss(const ValueType                  &mass_flux,
                                  [[maybe_unused]] const ValueType &temperature)
    {
      ValueType specific_enthalpy(0.0);

      if (do_phenomenological_recoil_pressure)
        {
          specific_enthalpy =
            specific_heat_capacity * (temperature - specific_enthalpy_reference_temperature);
        }

      return mass_flux * (latent_heat_of_evaporation + specific_enthalpy);
    }


    /**
     * derivative of specific enthalpy h(T) with respect to the temperature:
     *
     *  d h(T)
     * -------- = c_p^sl
     *    dT
     *
     * tangent of heat sink due to evaporation:
     *
     *  d q_s      .
     * ------- = - m * c_p^sl
     *    dT
     */
    template <typename ValueType>
    inline ValueType
    compute_evaporative_heat_loss_derivative([[maybe_unused]] const ValueType &mass_flux,
                                             [[maybe_unused]] const ValueType &temperature)
    {
      if (not do_phenomenological_recoil_pressure)
        return 0.0;

      return mass_flux * temperature * specific_heat_capacity;
    }

  private:
    const bool   do_phenomenological_recoil_pressure;
    const number latent_heat_of_evaporation;
    const number specific_heat_capacity;
    const number specific_enthalpy_reference_temperature;
  };
} // namespace MeltPoolDG::Evaporation