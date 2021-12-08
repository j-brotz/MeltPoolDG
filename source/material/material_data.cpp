#include <meltpooldg/material/material_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  MaterialData<number>::add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter("default material",
                      default_material,
                      "If this parameter is initialized, the liquid and solid material parameters "
                      "will be set to the default material parameters of the specified material.");
    prm.add_parameter(
      "material first capacity",
      first.capacity,
      "capacity of the primary material (in case of two-phases corresponding to level-set = -1)");
    prm.add_parameter(
      "material first conductivity",
      first.conductivity,
      "conductivity of the primary material (in case of two-phases corresponding to level-set = -1)");
    prm.add_parameter(
      "material first density",
      first.density,
      "density of the primary material (in case of two-phases corresponding to level-set = -1)");
    prm.add_parameter(
      "material first viscosity",
      first.viscosity,
      "viscosity of the primary material (in case of two-phases corresponding to level-set = -1)");
    prm.add_parameter("material second capacity",
                      second.capacity,
                      "capacity of the secondary material (level-set = 1)");
    prm.add_parameter("material second conductivity",
                      second.conductivity,
                      "conductivity of the secondary material (level-set = 1)");
    prm.add_parameter("material second density",
                      second.density,
                      "density of the secondary material (level-set = 1)");
    prm.add_parameter("material second viscosity",
                      second.viscosity,
                      "viscosity of the secondary material (level-set = 1)");
    prm.add_parameter("material solid capacity", solid.capacity, "capacity of the solid phase");
    prm.add_parameter("material solid conductivity",
                      solid.conductivity,
                      "conductivity of the solid phase");
    prm.add_parameter("material solid density", solid.density, "density of the solid phase");
    prm.add_parameter("material solid viscosity", solid.viscosity, "viscosity of the solid phase");
    prm.add_parameter("material melting point", melting_point, "Melting point (K)");
    prm.add_parameter("material solidus temperature", solidus_temperature, "Solidus temperature");
    prm.add_parameter("material liquidus temperature",
                      liquidus_temperature,
                      "Liquidus temperature");
    prm.add_parameter("material specific enthalpy reference temperature",
                      specific_enthalpy_reference_temperature,
                      "Reference temperature of the specific enthalpy");
    prm.add_parameter("material two phase properties transition type",
                      two_phase_properties_transition_type,
                      "Choose how to interpolate the properties over the interface.");
    prm.add_parameter("material boiling temperature",
                      boiling_temperature,
                      "Boiling temperature (K).");
    prm.add_parameter("material latent heat of evaporation",
                      latent_heat_of_evaporation,
                      "Latent heat of evaporation (J/kg).");
    prm.add_parameter("material molar mass", molar_mass, "Molar mass (mol/kg).");
    prm.add_parameter("material sticking constant", sticking_constant, "Sticking constant.");

    // set default material values if specified
    prm.add_action("default material", [&prm](const std::string &value) {
      DefaultMaterial value_enum = DefaultMaterial::not_initialized;
      try
        {
          value_enum = DefaultMaterial::_from_string(value.c_str());
        }
      catch (const std::runtime_error &)
        AssertThrow(false,
                    ExcMessage("\"" + value + "\" doesn't name a default material! Abort..."));

      switch (value_enum)
        {
          case DefaultMaterial::not_initialized:
            // nothing to do
            break;
            case DefaultMaterial::stainless_steel: {
              create_stainless_steel_material_data<number>().set_parameters(prm);
              break;
            }
            case DefaultMaterial::Ti64: {
              create_Ti64_material_data<number>().set_parameters(prm);
              break;
            }
          default:
            AssertThrow(false, ExcNotImplemented());
        }
    });
  }



  template <typename number>
  void
  MaterialData<number>::set_parameters(ParameterHandler &prm)
  {
    prm.set("default material", default_material);
    prm.set("material first capacity", first.capacity);
    prm.set("material first conductivity", first.conductivity);
    prm.set("material first density", first.density);
    prm.set("material first viscosity", first.viscosity);
    prm.set("material second capacity", second.capacity);
    prm.set("material second conductivity", second.conductivity);
    prm.set("material second density", second.density);
    prm.set("material second viscosity", second.viscosity);
    prm.set("material solid capacity", solid.capacity);
    prm.set("material solid conductivity", solid.conductivity);
    prm.set("material solid density", solid.density);
    prm.set("material solid viscosity", solid.viscosity);
    prm.set("material melting point", melting_point);
    prm.set("material solidus temperature", solidus_temperature);
    prm.set("material liquidus temperature", liquidus_temperature);
    prm.set("material specific enthalpy reference temperature",
            specific_enthalpy_reference_temperature);
    prm.set("material two phase properties transition type",
            two_phase_properties_transition_type._to_string());
    prm.set("material boiling temperature", boiling_temperature);
    prm.set("material latent heat of evaporation", latent_heat_of_evaporation);
    prm.set("material molar mass", molar_mass);
    prm.set("material sticking constant", sticking_constant);
  }



  template <typename number>
  MaterialData<number>
  create_stainless_steel_material_data()
  {
    MaterialData<number> data;
    data.first.capacity     = 10.0;   //  J / (kg K)
    data.first.conductivity = 0.026;  //  W / (m K)
    data.first.density      = 74.3;   //  kg / m³
    data.first.viscosity    = 6.0e-4; //  kg / (m s)
    // clang-format off
    data.second.capacity     = data.solid.capacity     = 965.0;  //  J / (kg K)
    data.second.conductivity = data.solid.conductivity = 35.95;  //  W / (m K)
    data.second.density      = data.solid.density      = 7430.0; //  kg / m³
    data.second.viscosity                              = 6.0e-3; //  kg / (m s)
    data.solidus_temperature = data.melting_point      = 1700.0; //  K
    // clang-format on
    data.boiling_temperature                     = 3000.0;  //  K
    data.latent_heat_of_evaporation              = 6.0e6;   //  J / kg
    data.molar_mass                              = 5.22e-2; //  kg / mol
    data.sticking_constant                       = 1.0;     //  dimensionless
    data.specific_enthalpy_reference_temperature = 663.731; //  K
    return data;
  }



  template <typename number>
  MaterialData<number>
  create_Ti64_material_data()
  {
    MaterialData<number> data;
    data.first.capacity     = 11.3;    //  J / (kg K)
    data.first.conductivity = 0.02863; //  W / (m K)
    data.first.density      = 44.1;    //  kg / m³
    data.first.viscosity    = 0.00035; //  kg / (m s)
    // clang-format off
    data.second.capacity     = data.solid.capacity     = 1130.0; //  J / (kg K)
    data.second.conductivity = data.solid.conductivity = 28.63;  //  W / (m K)
    data.second.density      = data.solid.density      = 4087.0; //  kg / m³
    data.second.viscosity                              = 0.0035; //  kg / (m s)
    data.solidus_temperature = data.melting_point      = 1933;   //  K
    // clang-format on
    data.boiling_temperature                     = 3133.0;  //  K
    data.latent_heat_of_evaporation              = 8.84e6;  //  J / kg
    data.molar_mass                              = 4.78e-2; //  kg / mol
    data.sticking_constant                       = 1.0;     //  dimensionless
    data.specific_enthalpy_reference_temperature = 538.0;   //  K
    return data;
  }



  template struct MaterialData<double>;
  template MaterialData<double>
  create_stainless_steel_material_data<double>();
  template MaterialData<double>
  create_Ti64_material_data<double>();
} // namespace MeltPoolDG