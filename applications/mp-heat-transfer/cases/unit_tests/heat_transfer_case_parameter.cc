#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../../heat_transfer_case.hpp"

int
main()
{
  dealii::ParameterHandler                             prm;
  MeltPoolDG::Heat::HeatTransferCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, false /*print_details*/);
  return 0;
}
