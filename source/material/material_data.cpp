#include <meltpooldg/material/material_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  MaterialData<number>::add_parameters(ParameterHandler &prm)
  {
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

  template struct MaterialData<double>;
} // namespace MeltPoolDG