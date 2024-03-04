#include <meltpooldg/interface/parameters.hpp>

#include <iostream>

int
main()
{
  dealii::ParameterHandler       prm;
  MeltPoolDG::Parameters<double> parameters;
  parameters.print_parameters(prm, std::cout, false /*print_details*/);
  return 0;
}
