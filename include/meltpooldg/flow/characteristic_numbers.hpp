/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, November 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/material/material.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  class CharacteristicNumbers
  {
  private:
    const MaterialParameterValues<number> &material;

  public:
    CharacteristicNumbers(const MaterialParameterValues<number> &material_phase);

    /**
     * Compute the Mach number Ma
     *
     *         ρ · c_p  · v · d
     *    Ma = -----------------
     *                 λ
     *
     * with:
     *   ρ   ... density (kg/m³)
     *   c_p ... heat capacity (J/(kgK))
     *   v   ... characteristic velocity (m/s)
     *   d   ... characteristic length (m)
     *   λ   ... heat conductivity (W/(mK))
     */
    number
    Mach(const number &characteristic_velocity, const number &characteristic_length) const;

    /**
     * Compute the Reynolds number Re
     *
     *         ρ · v · d
     *    Re = ---------
     *             µ
     *
     * with:
     *   ρ ... density (kg/m³)
     *   v ... characteristic velocity (m/s)
     *   d ... characteristic length (m)
     *   µ ... dynamic viscosity (Ns/m²)
     */
    number
    Reynolds(const number &characteristic_velocity, const number &characteristic_length) const;

    /**
     * Compute the capillary number Ca
     *
     *         µ · v
     *    Ca = -----
     *           σ
     *
     * with:
     *   µ ... dynamic viscosity (Ns/m²)
     *   v ... characteristic velocity (m/s)
     *   σ ... surface tension coefficient (N/m)
     */
    number
    capillary(const number &characteristic_velocity,
              const number &surface_tension_coefficient) const;
  };
} // namespace MeltPoolDG::Flow
