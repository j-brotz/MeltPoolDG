#pragma once

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
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
} // namespace MeltPoolDG
