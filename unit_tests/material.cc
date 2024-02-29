#include <deal.II/base/vectorization.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/material/material.templates.hpp>

#include <iomanip>
#include <iostream>

using namespace MeltPoolDG;
using namespace dealii;

const auto setw = std::setw(8);

void
print_material_parameter_values(const MaterialParameterValues<double> &values)
{
  std::cout << setw << values.thermal_conductivity << "|" << setw << values.specific_heat_capacity
            << "|" << setw << values.density << "|" << setw << values.dynamic_viscosity << "|"
            << setw << values.d_thermal_conductivity_d_T << "|" << setw
            << values.d_specific_heat_capacity_d_T << "|" << setw << values.d_density_d_T << "|"
            << setw << values.gas_fraction << "|" << setw << values.liquid_fraction << "|" << setw
            << values.solid_fraction << std::endl;
}

void
print_material_parameter_values(const MaterialParameterValues<VectorizedArray<double>> &values)
{
  std::cout << setw << values.thermal_conductivity[0] << "|" << setw
            << values.specific_heat_capacity[0] << "|" << setw << values.density[0] << "|" << setw
            << values.dynamic_viscosity[0] << "|" << setw << values.d_thermal_conductivity_d_T[0]
            << "|" << setw << values.d_specific_heat_capacity_d_T[0] << "|" << setw
            << values.d_density_d_T[0] << "|" << setw << values.gas_fraction[0] << "|" << setw
            << values.liquid_fraction[0] << "|" << setw << values.solid_fraction[0] << std::endl;
}

int
main()
{
  MaterialData<double> material_data;

  material_data.gas.thermal_conductivity   = 10.0;
  material_data.gas.specific_heat_capacity = 0.0;
  material_data.gas.density                = 1000.0;
  material_data.gas.dynamic_viscosity      = 1.0;

  material_data.liquid.thermal_conductivity   = 40.0;
  material_data.liquid.specific_heat_capacity = 1000.0;
  material_data.liquid.density                = 8000.0;
  material_data.liquid.dynamic_viscosity      = 100;

  material_data.solid.thermal_conductivity   = 30.0;
  material_data.solid.specific_heat_capacity = 900.0;
  material_data.solid.density                = 7000.0;
  material_data.solid.dynamic_viscosity      = 1000.0;

  material_data.solid_liquid_properties_transition_type =
    SolidLiquidPropertiesTransitionType::mushy_zone;
  material_data.two_phase_fluid_properties_transition_type =
    TwoPhaseFluidPropertiesTransitionType::smooth;
  material_data.solidus_temperature  = 0.0;
  material_data.liquidus_temperature = 100.0;

  const auto update_all =
    MaterialUpdateFlags::thermal_conductivity | MaterialUpdateFlags::specific_heat_capacity |
    MaterialUpdateFlags::density | MaterialUpdateFlags::d_thermal_conductivity_d_T |
    MaterialUpdateFlags::d_specific_heat_capacity_d_T | MaterialUpdateFlags::dynamic_viscosity |
    MaterialUpdateFlags::d_density_d_T | MaterialUpdateFlags::phase_fractions;

  // testing values
  const double                  ls_heaviside_val = 0.2;
  const double                  temperature_val  = 20.0;
  const VectorizedArray<double> ls_heaviside_vec(0.6);
  const VectorizedArray<double> temperature_vec(60.0);

  // test material types

  std::cout
    << "       k|      cp|     rho|      nu| d_k_d_T|d_cp_d_T|d_rho_dT|     X_g|     X_l|     X_s"
    << std::endl;
  std::cout << "single_phase" << std::endl;
  {
    const Material material(material_data, MaterialTypes::single_phase);

    const auto data_double = material.compute_parameters<double>(update_all);
    print_material_parameter_values(data_double);

    const auto data_vec = material.compute_parameters<VectorizedArray<double>>(update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "gas_liquid" << std::endl;
  {
    const Material material(material_data, MaterialTypes::gas_liquid);

    const auto data_double = material.compute_parameters<double>(ls_heaviside_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec =
      material.compute_parameters<VectorizedArray<double>>(ls_heaviside_vec, update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "gas_liquid_consistent_with_evaporation" << std::endl;
  {
    const Material material(material_data, MaterialTypes::gas_liquid_consistent_with_evaporation);

    const auto data_double = material.compute_parameters<double>(ls_heaviside_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec =
      material.compute_parameters<VectorizedArray<double>>(ls_heaviside_vec, update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "liquid_solid" << std::endl;
  {
    const Material material(material_data, MaterialTypes::liquid_solid);

    const auto data_double = material.compute_parameters<double>(temperature_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec =
      material.compute_parameters<VectorizedArray<double>>(temperature_vec, update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "gas_liquid_solid" << std::endl;
  {
    const Material material(material_data, MaterialTypes::gas_liquid_solid);

    const auto data_double =
      material.compute_parameters<double>(ls_heaviside_val, temperature_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec = material.compute_parameters<VectorizedArray<double>>(ls_heaviside_vec,
                                                                               temperature_vec,
                                                                               update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "gas_liquid_solid_consistent_with_evaporation" << std::endl;
  {
    const Material material(material_data,
                            MaterialTypes::gas_liquid_solid_consistent_with_evaporation);

    const auto data_double =
      material.compute_parameters<double>(ls_heaviside_val, temperature_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec = material.compute_parameters<VectorizedArray<double>>(ls_heaviside_vec,
                                                                               temperature_vec,
                                                                               update_all);
    print_material_parameter_values(data_vec);
  }

  // test functionalities
  const Material material(material_data, MaterialTypes::gas_liquid_solid);

  std::cout << "only update selected" << std::endl;
  const auto data_flow =
    material.compute_parameters<double>(ls_heaviside_val,
                                        temperature_val,
                                        MaterialUpdateFlags::density |
                                          MaterialUpdateFlags::dynamic_viscosity);
  print_material_parameter_values(data_flow);
}
