#include <deal.II/base/parameter_handler.h>

#include <iostream>

#include "../melt_pool_case.hpp"
int
main()
{
  dealii::ParameterHandler                   prm;
  MeltPoolDG::MeltPoolCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, false /*print_details*/);
  return 0;
}
