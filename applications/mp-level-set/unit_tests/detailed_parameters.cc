#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../level_set_case.hpp"

int
main()
{
  dealii::ParameterHandler                             prm;
  MeltPoolDG::LevelSet::LevelSetCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, true /*print_details*/);
  return 0;
}
