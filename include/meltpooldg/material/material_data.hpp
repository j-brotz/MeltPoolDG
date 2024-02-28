#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>

namespace MeltPoolDG
{
  BETTER_ENUM(
    MaterialTemplate,
    char,
    none,            // all material parameters must be specified
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
  struct MaterialPhaseData
  {
    number thermal_conductivity   = 0.0;
    number specific_heat_capacity = 0.0;
    number density                = 0.0;
    number dynamic_viscosity      = 0.0;

    void
    add_parameters(dealii::ParameterHandler &prm, const std::string &phase_name);
  };

  /**
   * Parameters of the Material class.
   *
   * @warning If you want to use a predefined material as template (by specifying
   * <material template>) but also modify individual properties, you need to
   * specify it in the first place in the <material> section.
   */
  template <typename number = double>
  struct MaterialData
  {
  private:
    MaterialTemplate material_template = MaterialTemplate::none;

  public:
    /**
     * Material parameters of the gas phase; heaviside(level set) == 0
     */
    MaterialPhaseData<number> gas;

    /**
     * Material parameters of the liquid phase; heaviside(level set) == 1
     */
    MaterialPhaseData<number> liquid;

    /**
     * Solid material.
     */
    MaterialPhaseData<number> solid;

    number solidus_temperature        = 0.0;
    number liquidus_temperature       = 0.0;
    number melting_point              = 0.0;
    number boiling_temperature        = 0.0;
    number latent_heat_of_evaporation = 0.0;
    number molar_mass                 = 0.0;
    number sticking_constant          = 1.0;

    number specific_enthalpy_reference_temperature = numbers::invalid_double;

    SolidLiquidPropertiesTransitionType solidification_type =
      SolidLiquidPropertiesTransitionType::sharp; // TODO rename parameter according to enum
    TwoPhaseFluidPropertiesTransitionType two_phase_properties_transition_type =
      TwoPhaseFluidPropertiesTransitionType::sharp; // TODO rename parameter according to enum  and
                                                    // set default value to smooth

    void
    add_parameters(dealii::ParameterHandler &prm);

    /**
     * Creates MaterialData with all parameters set to the values of stainless steel:
     *
     * gas capacity               = 10.0  J / (kg K)
     * gas conductivity           = 0.026  W / (m K)
     * gas density                = 74.3  kg / m³
     * gas viscosity              = 6.0e-4  kg / (m s)
     * liquid capacity            = 965  J / (kg K)
     * liquid conductivity        = 35.95  W / (m K)
     * liquid density             = 7430  kg / m³
     * liquid viscosity           = 6.0e-3  kg / (m s)
     * solid capacity             = 965  J / (kg K)
     * solid conductivity         = 35.95  W / (m K)
     * solid density              = 7430  kg / m³
     * solid viscosity            = 0.6  kg / (m s)
     * solidus temperature        = 1700  K
     * liquidus temperature       = 2100  K
     * melting point              = 1700  K
     * boiling temperature        = 3000  K
     * latent heat of evaporation = 6.0e6  J / kg
     * molar mass                 = 5.22e-2  kg / mol
     * sticking constant          = 1
     * specific enthalpy reference temperature = 663.731  K
     *
     * @note This function also sets these computational parameters as follows:
     *
     * solidification type                  = mushy zone
     * two phase properties transition type = smooth
     */
    static MaterialData<number>
    create_stainless_steel_material_data();

    /**
     * Creates MaterialData with all parameters set to the values of Ti-6Al-4V:
     *
     * gas capacity               = 11.3  J / (kg K)
     * gas conductivity           = 0.02863  W / (m K)
     * gas density                = 44.1  kg / m³
     * gas viscosity              = 0.00035  kg / (m s)
     * liquid capacity            = 1130  J / (kg K)
     * liquid conductivity        = 28.63  W / (m K)
     * liquid density             = 4087  kg / m³
     * liquid viscosity           = 0.0035  kg / (m s)
     * solid capacity             = 1130  J / (kg K)
     * solid conductivity         = 28.63  W / (m K)
     * solid density              = 4087  kg / m³
     * solid viscosity            = 0.35  kg / (m s)
     * solidus temperature        = 1933  K
     * liquidus temperature       = 2200  K
     * melting point              = 1933  K
     * boiling temperature        = 3133  K
     * latent heat of evaporation = 8.84e6  J / kg
     * molar mass                 = 4.78e-2  kg / mol
     * sticking constant          = 1
     * specific enthalpy reference temperature = 538  K
     *
     * @note This function also sets these computational parameters as follows:
     *
     * solidification type                  = mushy zone
     * two phase properties transition type = smooth
     */
    static MaterialData<number>
    create_Ti64_material_data();
  };
} // namespace MeltPoolDG
