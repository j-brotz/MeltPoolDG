#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <string>

namespace MeltPoolDG::LevelSet
{
  BETTER_ENUM(ConvectionStabilizationType,
              char,
              none,
              // streamline upwind Petrov-Galerkin stabilization
              SUPG)

  template <typename number = double>
  struct AdvectionDiffusionData
  {
    AdvectionDiffusionData();

    number      diffusivity             = 0.0;
    std::string time_integration_scheme = "crank_nicolson";
    std::string implementation          = "meltpooldg";

    struct ConvectionStabilizationData
    {
      ConvectionStabilizationType type        = ConvectionStabilizationType::none;
      double                      coefficient = -1.0;
    } conv_stab;

    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post();

    void
    check_input_parameters() const;
  };
} // namespace MeltPoolDG::LevelSet
