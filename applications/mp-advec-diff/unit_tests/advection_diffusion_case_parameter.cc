#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../advection_diffusion_case.hpp"

int
main()
{
  dealii::ParameterHandler                                       prm;
  MeltPoolDG::LevelSet::AdvectionDiffusionCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, false /*print_details*/);
  return 0;
}
