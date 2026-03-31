#include <meltpooldg/core/parameters_base.hpp>

#include <iostream>

#include "../radiative_transport_case.hpp"

int
main()
{
  dealii::ParameterHandler                                                 prm;
  MeltPoolDG::RadiativeTransport::RadiativeTransportCaseParameters<double> parameters;
  parameters.print_parameters(prm, std::cout, true /*print_details*/);
  return 0;
}
