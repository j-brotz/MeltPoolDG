#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../meltpool_vapor_flow_case.hpp"

int
main()
{
  dealii::ParameterHandler                              prm;
  MeltPoolDG::MeltPoolVaporFlow::CaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, false /*print_details*/);
  return 0;
}
