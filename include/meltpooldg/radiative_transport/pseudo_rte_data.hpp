#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

namespace MeltPoolDG::RadiativeTransport
{
  template <typename number>
  struct PseudoTimeSteppingData
  {
    PseudoTimeSteppingData();

    LinearSolverData<number>                  linear_solver;
    TimeIntegration::TimeSteppingData<number> time_stepping_data;
    number                                    diffusion_term_scaling = 1.;
    number                                    advection_term_scaling = 1.;
    number                                    pseudo_time_scaling    = 0.01;
    number                                    rel_tolerance          = 1e-3;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::RadiativeTransport