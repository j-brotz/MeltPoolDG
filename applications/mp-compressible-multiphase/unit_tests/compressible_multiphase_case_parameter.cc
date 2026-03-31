#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../compressible_multiphase_case.hpp"

int
main()
{
  dealii::ParameterHandler                                             prm;
  MeltPoolDG::Multiphase::CompressibleMultiphaseCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, false /*print_details*/);
  return 0;
}
