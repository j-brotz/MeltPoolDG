#include <meltpooldg/material/material_data.hpp>

#include <iostream>

// Test that parameters are read in order they are specified in the parameter file.

using namespace MeltPoolDG;

int
main()
{
  {
    MaterialData<double> material_data;
    ParameterHandler     prm;

    material_data.add_parameters(prm);

    std::istringstream is("{\"material template\" : \"stainless_steel\"}");
    prm.parse_input_from_json(is);

    AssertDimension(material_data.first.capacity,
                    MaterialData<double>::create_stainless_steel_material_data().first.capacity);

    std::cout << "OK!" << std::endl;
  }
  {
    MaterialData<double> material_data;
    ParameterHandler     prm;

    material_data.add_parameters(prm);

    std::istringstream is("{\"material first capacity\" : \"5\", "
                          "\"material template\" : \"stainless_steel\"}");
    prm.parse_input_from_json(is);

    AssertDimension(material_data.first.capacity,
                    MaterialData<double>::create_stainless_steel_material_data().first.capacity);

    std::cout << "OK!" << std::endl;
  }
  {
    MaterialData<double> material_data;
    ParameterHandler     prm;

    material_data.add_parameters(prm);

    std::istringstream is("{\"material template\" : \"stainless_steel\", "
                          "\"material first capacity\" : \"5\"}");
    prm.parse_input_from_json(is);

    AssertDimension(material_data.first.capacity, 5);

    std::cout << "OK!" << std::endl;
  }
}
