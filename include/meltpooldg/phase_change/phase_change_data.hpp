#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG::Multiphase
{
  /**
   * @brief Collection of parameters related to liquid-gas and solid-liquid phase transitions.
   */
  template <typename number>
  struct PhaseChangeData
  {
    /// Parameters for the liquid-gas phase change
    struct LiquidGas
    {
      /// Boiling temperature at given reference pressure (SI: K)
      number boiling_temperature = 3133.0;

      /// Reference pressure for boiling temperature (SI: Pa)
      number reference_pressure = 1.e5;

      /// Latent heat of vaporization (SI: J/kg)
      number latent_heat_of_vaporization = 8.84e6;
    } liquid_gas;

    /// Parameters for the solid-liquid phase change
    struct SolidLiquid
    {
      /// Optional darcy damping
      bool use_darcy_damping = false;

      /// Liquidus temperature (SI: K)
      number liquidus_temperature = 2200.;

      /// Solidus temperature (SI: K)
      number solidus_temperature = 1933.;
    } solid_liquid;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("phase change");
      {
        prm.enter_subsection("liquid gas");
        prm.add_parameter("boiling temperature",
                          liquid_gas.boiling_temperature,
                          "Boiling temperature at given reference pressure.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("reference pressure",
                          liquid_gas.reference_pressure,
                          "Reference pressure for boiling temperature.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("latent heat of vaporization",
                          liquid_gas.latent_heat_of_vaporization,
                          "Latent heat of vaporization (J/kg).",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.leave_subsection();
        prm.enter_subsection("solid liquid");
        prm.add_parameter("use darcy damping",
                          solid_liquid.use_darcy_damping,
                          "Use Darcy damping?",
                          dealii::Patterns::Bool());
        prm.add_parameter("liquidus temperature",
                          solid_liquid.liquidus_temperature,
                          "Liquidus temperature.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.add_parameter("solidus temperature",
                          solid_liquid.solidus_temperature,
                          "Solidus temperature.",
                          dealii::Patterns::Double(0., std::numeric_limits<number>::max()));
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG::Multiphase
