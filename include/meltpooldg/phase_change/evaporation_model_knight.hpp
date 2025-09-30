#pragma once

namespace MeltPoolDG::Evaporation
{
  /**
   * @brief This class implements the evaporative mass flux and temperature jump for rapid evaporation according to
   * Knight's theory.
   *
   * Knight, C. J. (1979). Theoretical modeling of rapid surface vaporization with back pressure.
   * AIAA journal, 17(5), 519-523. DOI: 10.2514/3.61164
   *
   * It is derived from kinetic gas theory and couples the liquid and gaseous phase in a
   * thermodynamically consistent way.
   *
   * @note Currently, a monoatomic ideal gas is assumed. In addition, the material is assumed to consist of one species
   * with averaged material properties. No component-wise evaporation behavior is considered for
   * alloys.
   */
  template <typename number>
  class EvaporationModelKnight
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param atmospheric_pressure Atmospheric pressure in the gas phase (Gas chamber pressure).
     * @param boiling_temperature_at_atmospheric_pressure Boiling temperature of the considered material at atmospheric
     * pressure conditions.
     * @param latent_heat_of_evaporation Latent heat of evaporation.
     * @param specific_gas_constant Specific gas constant.
     * @param specific_heat_ratio_vapor Ratio of specific heats for the vapor phase.
     */
    explicit EvaporationModelKnight(const number atmospheric_pressure,
                                    const number boiling_temperature_at_atmospheric_pressure,
                                    const number latent_heat_of_evaporation,
                                    const number specific_gas_constant,
                                    const number specific_heat_ratio_vapor);

    /**
     * @brief Use Knight's theory to evaluate the evaporative mass flux and temperature jump
     * [T_liquid - T_gas].
     *
     * @param T_liquid Current liquid surface temperature.
     * @param Ma_gas Current Mach number in the gas phase at interface position.
     *
     * @note This function just updates the current values for the evaporative mass flux and
     * temperature jump. Use the provided getter functions for access. Make sure to properly call
     * this function before you apply the getter functions for mass flux and temperature jump.
     */
    void
    local_evaluate_evaporative_mass_flux_and_temperature_jump(const number &T_liquid,
                                                              const number &Ma_gas);

    /**
     * @brief Getter function for the evaporative mass flux.
     *
     * @return Evaporative mass flux.
     */
    number
    get_evaporative_mass_flux() const
    {
      return evaporative_mass_flux;
    }

    /**
     * @brief Getter function for the temperature jump [T_liquid - T_gas].
     *
     * @return Temperature jump [T_liquid - T_gas].
     */
    number
    get_temperature_jump() const
    {
      return temperature_jump;
    }

  private:
    /// Atmospheric pressure in the gas phase (Gas chamber pressure)
    const number atmospheric_pressure;

    /// Boiling temperature of the considered material at atmospheric pressure conditions
    const number boiling_temperature_at_atmospheric_pressure;

    /// Latent heat of evaporation
    const number latent_heat_of_evaporation;

    /// Specific gas constant
    const number specific_gas_constant;

    /// Ratio of specific heats for the vapor phase
    const number specific_heat_ratio_vapor;

    /// Evaporative mass flux from Knight's theory
    number evaporative_mass_flux = 0.;

    /// Temperature jump from Knight's theory
    number temperature_jump = 0.;

    /**
     * @brief Calculate the saturated vapor pressure at a certain vapor temperature according to the
     * Clausius-Clapeyron equation.
     *
     * @param T_sat Satured vapor temperature.
     *
     * @return Satured vapor pressure at the provided saturated vapor temperature @p T_sat.
     */
    number
    compute_saturated_vapor_pressure_from_clausius_clapeyron(const number &T_sat) const;

    /**
     * @brief Calculate the temperature ratio T_gas/T_sat according to Knight's theory.
     *
     * @param m Dimensionless velocity.
     *
     * @return Temperature ratio T_gas/T_sat.
     */
    number
    compute_temperature_ratio(const number &m) const;

    /**
     * @brief Calculate the density ratio rho_gas/rho_sat according to Knight's theory.
     *
     * @param T_gas_over_T_sat Temperature ratio T_gas/T_sat.
     * @param m Dimensionless velocity.
     *
     * @return Density ratio rho_gas/rho_sat.
     */
    number
    compute_density_ratio(const number &T_gas_over_T_sat, const number &m) const;

    /**
     * @brief Calculate the factor beta according to Knight's theory.
     *
     * This factor corresponds to the amount of back-scattered atoms in the Knudsen layer.
     * It can be determined from kinetic gas theory.
     *
     * @param T_gas_over_T_sat Temperature ratio T_gas/T_sat.
     * @param rho_gas_over_rho_sat Density ratio rho_gas/rho_sat.
     * @param m Dimensionless velocity.
     *
     * @return Factor beta.
     */
    number
    compute_factor_beta(const number &T_gas_over_T_sat,
                        const number &rho_gas_over_rho_sat,
                        const number &m) const;
  };
} // namespace MeltPoolDG::Evaporation
