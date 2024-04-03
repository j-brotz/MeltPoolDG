#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/reinitialization/reinitialization_data.hpp>

#include <string>

namespace MeltPoolDG::LevelSet
{

  template <typename number = double>
  struct CurvatureData
  {
    CurvatureData();

    bool         enable                  = true;
    number       filter_parameter        = 2.0;
    std::string  implementation          = "meltpooldg";
    unsigned int verbosity_level         = 0;
    bool         do_curvature_correction = false;

    struct NarrowBand
    {
      bool   enable              = false;
      number level_set_threshold = 0.9999999;
    } narrow_band;
    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const InterfaceThicknessParameterType &type) const;

    void
    post();
  };
} // namespace MeltPoolDG::LevelSet
