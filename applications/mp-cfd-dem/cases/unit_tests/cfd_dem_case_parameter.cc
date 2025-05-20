#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../../cfd_dem_case.hpp"

int
main()
{
  dealii::ParameterHandler                 prm;
  MeltPoolDG::CfdDemCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, false);
  return 0;
}
