#include <deal.II/base/exceptions.h>

#include <meltpooldg/material/material_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  MaterialPhaseData<number>::add_parameters(dealii::ParameterHandler &prm,
                                            const std::string        &phase_name)
  {
    prm.enter_subsection(phase_name);
    {
      prm.add_parameter("thermal conductivity",
                        thermal_conductivity,
                        "thermal conductivity of the " + phase_name + " phase");
      prm.add_parameter("specific heat capacity",
                        specific_heat_capacity,
                        "specific heat capacity of the " + phase_name + " phase");
      prm.add_parameter("density", density, "density of the " + phase_name + " phase");
      prm.add_parameter("dynamic viscosity",
                        dynamic_viscosity,
                        "dynamic viscosity of the " + phase_name + " phase");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  MaterialData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("material");
    {
      prm.add_parameter("material template",
                        material_template,
                        "If this parameter is initialized, the material parameters "
                        "of the specified material will be used as template. Individual "
                        "properties can be modified. However, be aware to put "
                        "<material template> in the first place of the <material> "
                        "section in these cases.");

      if (material_template == MaterialTemplate::none)
        prm.add_action("material template", [this](const std::string &value) {
          switch (MaterialTemplate::_from_string(value.c_str()))
            {
              case MaterialTemplate::none:
                // nothing to do
                break;
                case MaterialTemplate::stainless_steel: {
                  *this = create_stainless_steel_material_data();
                  break;
                }
                case MaterialTemplate::Ti64: {
                  *this = create_Ti64_material_data();
                  break;
                }
              default:
                AssertThrow(false, dealii::ExcNotImplemented());
            }
        });

      gas.add_parameters(prm, "gas");
      liquid.add_parameters(prm, "liquid");
      solid.add_parameters(prm, "solid");

      prm.add_parameter("solidus temperature", solidus_temperature, "Solidus temperature (K).");
      prm.add_parameter("liquidus temperature", liquidus_temperature, "Liquidus temperature (K).");
      prm.add_parameter("boiling temperature", boiling_temperature, "Boiling temperature (K).");
      prm.add_parameter("latent heat of evaporation",
                        latent_heat_of_evaporation,
                        "Latent heat of evaporation (J/kg).");
      prm.add_parameter("molar mass", molar_mass, "Molar mass (mol/kg).");
      prm.add_parameter("specific enthalpy reference temperature",
                        specific_enthalpy_reference_temperature,
                        "Reference temperature of the specific enthalpy");

      prm.add_parameter(
        "two phase fluid properties transition type",
        two_phase_fluid_properties_transition_type,
        "Choose how to interpolate the properties over the interface. "
        "sharp: properties jump at heaviside = 0.5; "
        "smooth: properties are smeared between the phases proportional to the heaviside (default); "
        "consistent_with_evaporation: same as \"smooth\", but the density is interpolated proportional by the harmonic mean.");
      prm.add_parameter(
        "solid liquid properties transition type",
        solid_liquid_properties_transition_type,
        "Choose how to interpolate the properties over between the liquid and the solid phase. "
        "mushy_zone: solid and liquid properties are interpolated between the solidus and liquidus temperature (default); "
        "sharp: the solid and liquid properties jump at the melting point, which is set via the solidus temperature.");
    }
    prm.leave_subsection();
  }



  template <typename number>
  MaterialData<number>
  MaterialData<number>::create_stainless_steel_material_data()
  {
    MaterialData<number> data;
    data.material_template = MaterialTemplate::stainless_steel;

    data.gas.thermal_conductivity   = 0.026;  //  W / (m K)
    data.gas.specific_heat_capacity = 10.0;   //  J / (kg K)
    data.gas.density                = 74.3;   //  kg / m³
    data.gas.dynamic_viscosity      = 6.0e-4; //  kg / (m s)
    // clang-format off
    data.liquid.thermal_conductivity   = data.solid.thermal_conductivity   = 35.95;  //  W / (m K)
    data.liquid.specific_heat_capacity = data.solid.specific_heat_capacity = 965.0;  //  J / (kg K)
    data.liquid.density                = data.solid.density                = 7430.0; //  kg / m³
    data.liquid.dynamic_viscosity                                          = 6.0e-3; //  kg / (m s)
    data.solid.dynamic_viscosity                                           = 0.6;    //  kg / (m s)
    // clang-format on
    data.solidus_temperature                     = 1700.0;  //  K
    data.liquidus_temperature                    = 2100.0;  //  K
    data.boiling_temperature                     = 3000.0;  //  K
    data.latent_heat_of_evaporation              = 6.0e6;   //  J / kg
    data.molar_mass                              = 5.22e-2; //  kg / mol
    data.specific_enthalpy_reference_temperature = 663.731; //  K
    return data;
  }



  template <typename number>
  MaterialData<number>
  MaterialData<number>::create_Ti64_material_data()
  {
    MaterialData<number> data;
    data.material_template = MaterialTemplate::Ti64;

    data.gas.thermal_conductivity   = 0.02863; //  W / (m K)
    data.gas.specific_heat_capacity = 11.3;    //  J / (kg K)
    data.gas.density                = 44.1;    //  kg / m³
    data.gas.dynamic_viscosity      = 0.00035; //  kg / (m s)
    // clang-format off
    data.liquid.thermal_conductivity   = data.solid.thermal_conductivity   = 28.63;  //  W / (m K)
    data.liquid.specific_heat_capacity = data.solid.specific_heat_capacity = 1130.0; //  J / (kg K)
    data.liquid.density                = data.solid.density                = 4087.0; //  kg / m³
    data.liquid.dynamic_viscosity                                          = 0.0035; //  kg / (m s)
    data.solid.dynamic_viscosity                                           = 0.35;   //  kg / (m s)
    // clang-format on
    data.solidus_temperature                     = 1933;    //  K
    data.liquidus_temperature                    = 2200.0;  //  K
    data.boiling_temperature                     = 3133.0;  //  K
    data.latent_heat_of_evaporation              = 8.84e6;  //  J / kg
    data.molar_mass                              = 4.78e-2; //  kg / mol
    data.specific_enthalpy_reference_temperature = 538.0;   //  K
    return data;
  }


  template struct MaterialPhaseData<double>;
  template struct MaterialData<double>;
} // namespace MeltPoolDG
