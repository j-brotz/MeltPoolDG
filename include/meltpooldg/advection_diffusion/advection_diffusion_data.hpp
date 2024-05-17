#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/interface/finite_element_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/time_integration/time_integration_setup.hpp>
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

    number            diffusivity             = 0.0;
    TimeIntegrators   time_integration_scheme = TimeIntegrators::crank_nicolson;
    std::string       implementation          = "meltpooldg";
    FiniteElementData fe;

    bool enable_time_dependent_bc = false;

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
    post(const FiniteElementData &base_fe_data);

    void
    check_input_parameters() const;
  };
} // namespace MeltPoolDG::LevelSet
