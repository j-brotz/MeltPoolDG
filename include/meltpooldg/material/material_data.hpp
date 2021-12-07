#pragma once

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  BETTER_ENUM(
    DefaultMaterial,
    char,
    not_initialized, // all material parameters must be specified
    stainless_steel, // melt and solid material parameters will be set to stainless steel values
    Ti64             // melt and solid material parameters will be set to Ti-6Al-4V values
  )
  BETTER_ENUM(SolidLiquidPropertiesTransitionType,
              char,
              not_initialized,
              mushy_zone, // the liquid and solid properties are smeared between the liquidus and
                          // solididus temperature
              sharp       // the liquid and solid properties jump at melting temperature
  )
  BETTER_ENUM(
    TwoPhaseFluidPropertiesTransitionType,
    char,
    not_initialized,
    sharp,  // properties jump at level-set = 0
    smooth, // properties are smeared between the phases with the factor level-set-as-heaviside
    consistent_with_evaporation // the density is smeared between the phases consistent with the
                                // evaporation formulation
  )

  template <typename number = double>
  struct MaterialData
  {
    DefaultMaterial default_material = DefaultMaterial::not_initialized;
    /**
     * Default material. In case of two-phase flow; heaviside(level set) == 0
     */
    struct First
    {
      number capacity     = 0.0;
      number conductivity = 0.0;
      number density      = 0.0;
      number viscosity    = 0.0;
    } first;

    /**
     * Secondary material. In case of two-phase-flow; heaviside(level set) == 1
     */
    struct Second
    {
      number capacity     = 0.0;
      number conductivity = 0.0;
      number density      = 0.0;
      number viscosity    = 0.0;
    } second;

    /**
     * Solid material.
     */
    struct Solid
    {
      number capacity     = 0.0;
      number conductivity = 0.0;
      number density      = 0.0;
      number viscosity    = 0.0;
    } solid;

    number solidus_temperature        = 0.0;
    number liquidus_temperature       = 0.0;
    number melting_point              = 0.0;
    number inv_mushy_interval         = 0.0;
    number boiling_temperature        = 0.0;
    number latent_heat_of_evaporation = 0.0;
    number molar_mass                 = 0.0;
    number sticking_constant          = 1.0;

    number specific_enthalpy_reference_temperature = 0.0;

    SolidLiquidPropertiesTransitionType solidification_type =
      SolidLiquidPropertiesTransitionType::sharp; // TODO rename parameter according to enum
    TwoPhaseFluidPropertiesTransitionType two_phase_properties_transition_type =
      TwoPhaseFluidPropertiesTransitionType::sharp; // TODO rename parameter according to enum

    void
    add_parameters(ParameterHandler &prm);
  };

  /**
   * Sets the liquid's and solid's material parameters to stainless steel values:
   *
   * liquid capacity            = 965  J / (kg K)
   * liquid conductivity        = 35.95  W / (m K)
   * liquid density             = 7430  kg / m³
   * liquid viscosity           = 6.0e-3  kg / (m s)
   * solid capacity             = 965  J / (kg K)
   * solid conductivity         = 35.95  W / (m K)
   * solid density              = 7430  kg / m³
   * solidus temperature        = 1700  K
   * melting point              = 1700  K
   * boiling temperature        = 3000  K
   * latent heat of evaporation = 6.0e6  J / kg
   * molar mass                 = 5.22e-2  kg / mol
   * sticking constant          = 1
   * specific enthalpy reference temperature = 663.731  K
   */
  template <typename number>
  void
  create_stainless_steel_material_data(MaterialData<number> &);

  /**
   * Sets the liquid's and solid's material parameters to Ti-6Al-4V values:
   *
   * liquid capacity            = 1130  J / (kg K)
   * liquid conductivity        = 28.63  W / (m K)
   * liquid density             = 4087  kg / m³
   * liquid viscosity           = 0.0035  kg / (m s)
   * solid capacity             = 1130  J / (kg K)
   * solid conductivity         = 28.63  W / (m K)
   * solid density              = 4087  kg / m³
   * solidus temperature        = 1933  K
   * melting point              = 1933  K
   * boiling temperature        = 3133  K
   * latent heat of evaporation = 8.84e6  J / kg
   * molar mass                 = 4.78e-2  kg / mol
   * sticking constant          = 1
   * specific enthalpy reference temperature = 538  K
   */
  template <typename number>
  void
  create_Ti64_material_data(MaterialData<number> &);
} // namespace MeltPoolDG
