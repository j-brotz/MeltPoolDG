#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/material/material_data.hpp>

#include <iostream>
#include <sstream>

// Test that parameters are read in order they are specified in the parameter file.

using namespace MeltPoolDG;

int
main()
{
  {
    MaterialData<double> material_data;
    ParameterHandler     prm;

    material_data.add_parameters(prm);

    std::istringstream is("{\"material\": {\"material template\": \"stainless_steel\"}}");
    prm.parse_input_from_json(is);

    AssertDimension(
      material_data.gas.specific_heat_capacity,
      MaterialData<double>::create_stainless_steel_material_data().gas.specific_heat_capacity);

    std::cout << "OK!" << std::endl;
  }
  {
    MaterialData<double> material_data;
    ParameterHandler     prm;

    material_data.add_parameters(prm);

    std::istringstream is("{\"material\": {\"gas\": {\"specific heat capacity\": \"5\"}, "
                          "\"material template\": \"stainless_steel\"}}");
    prm.parse_input_from_json(is);

    AssertDimension(
      material_data.gas.specific_heat_capacity,
      MaterialData<double>::create_stainless_steel_material_data().gas.specific_heat_capacity);

    std::cout << "OK!" << std::endl;
  }
  {
    MaterialData<double> material_data;
    ParameterHandler     prm;

    material_data.add_parameters(prm);

    std::istringstream is("{\"material\": {\"material template\": \"stainless_steel\", "
                          "\"gas\": {\"specific heat capacity\": \"5\"}}}");
    prm.parse_input_from_json(is);

    AssertDimension(material_data.gas.specific_heat_capacity, 5);

    std::cout << "OK!" << std::endl;
  }
}
