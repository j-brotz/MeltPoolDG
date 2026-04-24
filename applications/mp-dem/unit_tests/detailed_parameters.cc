#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../dem_case.hpp"

int
main()
{
  dealii::ParameterHandler              prm;
  MeltPoolDG::DemCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, true);
  return 0;
}
