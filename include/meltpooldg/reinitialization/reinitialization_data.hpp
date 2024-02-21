#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>

#include <string>

namespace MeltPoolDG::Reinitialization
{
  template <typename number = double>
  struct ReinitializationData
  {
    ReinitializationData();

    unsigned int             max_n_steps          = 5; //@todo: move to LevelSetData
    number                   constant_epsilon     = -1.0;
    number                   scale_factor_epsilon = 0.5;
    std::string              modeltype            = "olsson2007";
    std::string              implementation       = "meltpooldg";
    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::Reinitialization