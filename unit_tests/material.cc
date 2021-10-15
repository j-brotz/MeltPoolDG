#include <deal.II/base/vectorization.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/material/material.hpp>

#include <iostream>

using namespace MeltPoolDG;
using namespace dealii;

const auto setw = std::setw(8);

void
print_material_parameter_values(const MaterialParameterValues<double> &values)
{
  std::cout << setw << values.capacity << "|" << setw << values.conductivity << "|" << setw
            << values.density << "|" << setw << values.viscosity << "|" << setw
            << values.d_capacity_d_T << "|" << setw << values.d_conductivity_d_T << "|" << setw
            << values.d_density_d_T << "|" << setw << values.gas_fraction << "|" << setw
            << values.liquid_fraction << "|" << setw << values.solid_fraction << std::endl;
}

void
print_material_parameter_values(const MaterialParameterValues<VectorizedArray<double>> &values)
{
  std::cout << setw << values.capacity[0] << "|" << setw << values.conductivity[0] << "|" << setw
            << values.density[0] << "|" << setw << values.viscosity[0] << "|" << setw
            << values.d_capacity_d_T[0] << "|" << setw << values.d_conductivity_d_T[0] << "|"
            << setw << values.d_density_d_T[0] << "|" << setw << values.gas_fraction[0] << "|"
            << setw << values.liquid_fraction[0] << "|" << setw << values.solid_fraction[0]
            << std::endl;
}

int
main()
{
  MaterialData<double> material_data;

  material_data.first.capacity     = 0.0;
  material_data.first.conductivity = 10.0;
  material_data.first.density      = 1000.0;
  material_data.first.viscosity    = 1.0;

  material_data.second.capacity     = 1000.0;
  material_data.second.conductivity = 40.0;
  material_data.second.density      = 8000.0;
  material_data.second.viscosity    = 100;

  material_data.solid.capacity     = 900.0;
  material_data.solid.conductivity = 30.0;
  material_data.solid.density      = 7000.0;
  material_data.solid.viscosity    = 1000.0;

  material_data.solidification_type  = SolidLiquidPropertiesTransitionType::mushy_zone;
  material_data.solidus_temperature  = 0.0;
  material_data.liquidus_temperature = 100.0;
  material_data.inv_mushy_interval =
    1 / (material_data.liquidus_temperature - material_data.solidus_temperature);

  const auto update_all = MaterialUpdateFlags::capacity | MaterialUpdateFlags::conductivity |
                          MaterialUpdateFlags::density | MaterialUpdateFlags::d_capacity_d_T |
                          MaterialUpdateFlags::viscosity | MaterialUpdateFlags::d_conductivity_d_T |
                          MaterialUpdateFlags::d_density_d_T | MaterialUpdateFlags::phase_fractions;

  // testing values
  const double                  ls_heaviside_val = 0.2;
  const double                  temperature_val  = 20.0;
  const VectorizedArray<double> ls_heaviside_vec(0.6);
  const VectorizedArray<double> temperature_vec(60.0);

  // test material types

  std::cout
    << "      cp|       k|     rho|      nu|d_cp_d_T| d_k_d_T|d_rho_dT|     X_g|     X_l|     X_s"
    << std::endl;
  std::cout << "single phase material" << std::endl;
  {
    const Material material(material_data, MaterialTypes::single_phase);

    const auto data_double = material.compute_parameters<double>(update_all);
    print_material_parameter_values(data_double);

    const auto data_vec = material.compute_parameters<VectorizedArray<double>>(update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "two phase material" << std::endl;
  {
    const Material material(material_data, MaterialTypes::gas_liquid);

    const auto data_double = material.compute_parameters<double>(ls_heaviside_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec =
      material.compute_parameters<VectorizedArray<double>>(ls_heaviside_vec, update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "two phase material consistent with evaporation" << std::endl;
  {
    const Material material(material_data, MaterialTypes::gas_liquid_consistent_with_evaporation);

    const auto data_double = material.compute_parameters<double>(ls_heaviside_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec =
      material.compute_parameters<VectorizedArray<double>>(ls_heaviside_vec, update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "solidification material" << std::endl;
  {
    const Material material(material_data, MaterialTypes::liquid_solid);

    const auto data_double = material.compute_parameters<double>(temperature_val, update_all);
    print_material_parameter_values(data_double);

    const auto data_vec =
      material.compute_parameters<VectorizedArray<double>>(temperature_vec, update_all);
    print_material_parameter_values(data_vec);
  }

  std::cout << "three phase material" << std::endl;
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

  std::cout << "three phase material consistent with evaporation" << std::endl;
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
  const auto data_flow = material.compute_parameters<double>(ls_heaviside_val,
                                                             temperature_val,
                                                             MaterialUpdateFlags::density |
                                                               MaterialUpdateFlags::viscosity);
  print_material_parameter_values(data_flow);
}
