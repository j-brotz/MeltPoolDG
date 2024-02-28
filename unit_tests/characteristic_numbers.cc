#include <meltpooldg/flow/characteristic_numbers.hpp>
#include <meltpooldg/material/material_data.hpp>

#include <iostream>

using namespace MeltPoolDG;

int
main()
{
  MaterialParameterValues<double> material_data;

  // parameter values taken from Meier et al. (2020)
  material_data.thermal_conductivity   = 2.4e-6;
  material_data.specific_heat_capacity = 1e-4;
  material_data.density                = 500.0;
  material_data.dynamic_viscosity      = 0.024;

  // testing values
  const double surface_tension_coefficient = 0.01;
  const double characteristic_velocity     = 0.024;
  const double characteristic_length       = 1.44e-3;

  // run test
  auto         characteristic_numbers = Flow::CharacteristicNumbers<double>(material_data);
  const double reynolds_number =
    characteristic_numbers.Reynolds(characteristic_velocity, characteristic_length);
  const double mach_number =
    characteristic_numbers.Mach(characteristic_velocity, characteristic_length);
  const double capillary_number =
    characteristic_numbers.capillary(characteristic_velocity, surface_tension_coefficient);

  std::cout << "Reynolds number: " << reynolds_number << std::endl;
  std::cout << "Mach number: " << mach_number << std::endl;
  std::cout << "Capillary number: " << capillary_number << std::endl;
}
