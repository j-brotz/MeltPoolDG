#include <meltpooldg/material/material_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  MaterialData<number>::add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter(
      "default material",
      default_material,
      "If this parameter is initialized, the liquid and solid material parameters "
      "will be overwritten by the default material parameters of the specified material.",
      Patterns::Selection("not_initialized|stainless_steel|Ti64"));
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
  }

  template <typename number>
  void
  MaterialData<number>::set_default_material_parameters()
  {
    switch (default_material)
      {
          case DefaultMaterial::not_initialized: {
            return;
          }
          case DefaultMaterial::stainless_steel: {
            // clang-format off
            second.capacity     = solid.capacity     = 965.0;  //  J / (kg K)
            second.conductivity = solid.conductivity = 35.95;  //  W / (m K)
            second.density      = solid.density      = 7430.0; //  kg / m³
            second.viscosity                         = 6.0e-3; //  kg / (m s)
            solidus_temperature = melting_point      = 1700.0; //  K
            // clang-format on
            boiling_temperature                     = 3000.0;  //  K
            latent_heat_of_evaporation              = 6.0e6;   //  J / kg
            molar_mass                              = 5.22e-2; //  kg / mol
            sticking_constant                       = 1.0;
            specific_enthalpy_reference_temperature = 663.731; //  K

            // TODO set surface tension to 1.8 N/m here, too
            return;
          }
          case DefaultMaterial::Ti64: {
            // clang-format off
            second.capacity     = solid.capacity     = 1130.0; //  J / (kg K)
            second.conductivity = solid.conductivity = 28.63;  //  W / (m K)
            second.density      = solid.density      = 4087.0; //  kg / m³
            second.viscosity                         = 0.0035; //  kg / (m s)
            solidus_temperature = melting_point      = 1933;   //  K
            // clang-format on
            boiling_temperature                     = 3133.0;  //  K
            latent_heat_of_evaporation              = 8.84e6;  //  J / kg
            molar_mass                              = 4.78e-2; //  kg / mol
            sticking_constant                       = 1.0;
            specific_enthalpy_reference_temperature = 538.0; //  K

            // TODO set surface tension to 1.493 N/m here, too
            return;
          }
      }
    Assert(false, ExcNotImplemented());
  }

  template struct MaterialData<double>;
} // namespace MeltPoolDG