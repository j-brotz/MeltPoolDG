#pragma once
#include <meltpooldg/core/material.hpp>

namespace MeltPoolDG::Flow
{
  /**
   * @brief Computes dimensionless characteristic numbers for fluid flow.
   *
   * This class provides methods to compute key dimensionless numbers used in
   * fluid dynamics and heat transfer, including the Mach number (Ma),
   * Reynolds number (Re), and Capillary number (Ca).
   */
  template <typename number>
  class CharacteristicNumbers
  {
  public:
    /**
     * @brief Constructor.
     *
     * Initializes the characteristic number calculator class with the relevant material parameters.
     *
     * @param material_phase Reference to a MaterialParameterValues object containing the relevant
     * physical properties.
     */
    CharacteristicNumbers(const MaterialParameterValues<number, number> &material_phase);

    /**
     * @brief Compute the Mach number Ma.
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
     *
     * @param characteristic_velocity Characteristic velocity of the considered simulation setup.
     * @param characteristic_length Characteristic length of the considered simulation setup.
     *
     * @return The computed Mach number.
     */
    number
    Mach(const number &characteristic_velocity, const number &characteristic_length) const;

    /**
     * @brief Compute the Reynolds number Re.
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
     *
     * @param characteristic_velocity Characteristic velocity of the considered simulation setup.
     * @param characteristic_length Characteristic length of the considered simulation setup.
     *
     * @return The computed Reynolds number.
     */
    number
    Reynolds(const number &characteristic_velocity, const number &characteristic_length) const;

    /**
     * @brief Compute the capillary number Ca.
     *
     *         µ · v
     *    Ca = -----
     *           σ
     *
     * with:
     *   µ ... dynamic viscosity (Ns/m²)
     *   v ... characteristic velocity (m/s)
     *   σ ... surface tension coefficient (N/m)
     *
     * @param characteristic_velocity Characteristic velocity of the considered simulation setup.
     * @param surface_tension_coefficient Provided surface tension coefficient.
     *
     * @return The computed capillary number.
     */
    number
    capillary(const number &characteristic_velocity,
              const number &surface_tension_coefficient) const;

  private:
    /// Material parameter struct which provides all relevant physical properties
    const MaterialParameterValues<number, number> &material;
  };
} // namespace MeltPoolDG::Flow
