#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

namespace MeltPoolDG::RadiativeTransport
{
  template <typename number = double>
  struct PseudoTimeSteppingData
  {
    PseudoTimeSteppingData();

    LinearSolverData<number> linear_solver;
    TimeSteppingData<number> time_stepping_data;
    double                   diffusion_term_scaling = 1.;
    double                   advection_term_scaling = 1.;
    double                   pseudo_time_scaling    = 0.01;
    double                   rel_tolerance          = 1e-3;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::RadiativeTransport