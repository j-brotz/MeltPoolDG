#pragma once

#include <deal.II/base/vectorization.h>

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
   * @tparam number Floating point type.
   * @tparam number_2 A vectorized array data type can be passed as second template parameter, that
   * is used for vectorized computation.
   *
   * @note By default, @p number_2 is set to the given floating point type @p number.
   *
   * @note Currently, a monoatomic ideal gas is assumed. In addition, the material is assumed to consist of one species
   * with averaged material properties. No component-wise evaporation behavior is considered for
   * alloys.
   */
  template <typename number, typename number_2 = number>
  class EvaporationModelKnight
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param atmospheric_pressure Atmospheric pressure in the gas phase (gas chamber pressure).
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
     * @brief Use Knight's theory to evaluate the current evaporative mass flux and temperature jump
     * T_liquid - T_gas.
     *
     * @param T_liquid Current liquid surface temperature.
     * @param Ma_gas Current Mach number in the gas phase at interface position.
     * @param n_active_lanes Number of active lanes in the VectorizedArray.
     *
     * @note This function just updates the current values for the evaporative mass flux and
     * temperature jump. Use the provided getter functions for access. Make sure to properly call
     * this function before you apply the getter functions for mass flux and temperature jump.
     */
    void
    reinit(const number_2    &T_liquid,
           const number_2    &Ma_gas,
           const unsigned int n_active_lanes = dealii::VectorizedArray<number>::size());

    /**
     * @brief Getter function for the evaporative mass flux.
     *
     * @return Evaporative mass flux.
     */
    [[nodiscard]] number_2
    get_evaporative_mass_flux() const
    {
      return evaporative_mass_flux;
    }

    /**
     * @brief Getter function for the temperature jump T_liquid - T_gas.
     *
     * @return Temperature jump T_liquid - T_gas.
     */
    [[nodiscard]] number_2
    get_temperature_jump() const
    {
      return temperature_jump;
    }

  private:
    /// Atmospheric pressure in the gas phase (gas chamber pressure)
    const number atmospheric_pressure;

    /// Boiling temperature of the considered material at atmospheric pressure conditions
    const number boiling_temperature_at_atmospheric_pressure;

    /// Latent heat of evaporation
    const number latent_heat_of_evaporation;

    /// Specific gas constant
    const number specific_gas_constant;

    /// Ratio of specific heats for the vapor phase
    const number specific_heat_ratio_vapor;

    /// Temperature constant for Clausius-Clapeyron equation
    const number temperature_constant;

    /// Precomputed helper variable for the mass flux computation
    const number helper_mass_flux;

    /// Precomputed helper variable for the temperature ratio computation
    const number helper_temperature_ratio;

    /// Evaporative mass flux from Knight's theory
    number_2 evaporative_mass_flux{};

    /// Temperature jump from Knight's theory
    number_2 temperature_jump{};

    /**
     * @brief Calculate the temperature ratio T_gas/T_sat according to Knight's theory.
     *
     * @param m_g Dimensionless gas velocity.
     *
     * @return Temperature ratio T_gas/T_sat.
     */
    number_2
    compute_temperature_ratio(const number_2 &m_g) const;

    /**
     * @brief Calculate the density ratio rho_gas/rho_sat according to Knight's theory.
     *
     * @param T_gas_over_T_sat Temperature ratio T_gas/T_sat.
     * @param m_g Dimensionless gas velocity.
     * @param exp_m_g_2 Precomputed exponential of the squared dimensionless gas velocity @p m_g.
     * @param erfc_m_g Precomputed complementary error function of the dimensionless gas velocity @p m_g.
     *
     * @return Density ratio rho_gas/rho_sat.
     */
    number_2
    compute_density_ratio(const number_2 &T_gas_over_T_sat,
                          const number_2 &m_g,
                          const number_2 &exp_m_g_2,
                          const number_2 &erfc_m_g) const;

    /**
     * @brief Calculate the factor beta according to Knight's theory.
     *
     * This factor corresponds to the amount of back-scattered atoms in the Knudsen layer.
     * It can be determined from kinetic gas theory.
     *
     * @param T_gas_over_T_sat Temperature ratio T_gas/T_sat.
     * @param rho_gas_over_rho_sat Density ratio rho_gas/rho_sat.
     * @param m_g Dimensionless velocity.
     * @param exp_m_g_2 Precomputed exponential of the squared dimensionless velocity @p m_g.
     *
     * @return Factor beta.
     */
    number_2
    compute_factor_beta(const number_2 &T_gas_over_T_sat,
                        const number_2 &rho_gas_over_rho_sat,
                        const number_2 &m_g,
                        const number_2 &exp_m_g_2) const;

    /**
     * @brief Restrict evaporation to liquid temperatures above the boiling temperature.
     *
     * This function simply sets the @p evaporative_mass_flux and @p temperature_jump to zero, if the
     * liquid surface temperature @p T_sat is below the boiling temperature.
     *
     * @param T_sat Saturated vapor temperature at liquid surface (=liquid surface temperature).
     */
    void
    apply_boiling_threshold(const number_2 &T_sat);
  };
} // namespace MeltPoolDG::Evaporation
