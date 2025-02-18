#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/level_set/reinitialization_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>

#include <string>

namespace MeltPoolDG::LevelSet
{

  template <typename number = double>
  struct CurvatureData
  {
    CurvatureData();

    bool        enable                  = true;
    number      filter_parameter        = 2.0;
    std::string implementation          = "meltpooldg";
    int         verbosity_level         = -1;
    bool        do_curvature_correction = false;

    struct NarrowBand
    {
      bool   enable              = false;
      number level_set_threshold = 0.9999999;
    } narrow_band;

    struct CurvatureDGSpecificData
    {
      number penalty_factor = 100.0;
    } curvature_DG_specific_data;

    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const unsigned int base_verbosity_level);

    void
    check_input_parameters(const InterfaceThicknessParameterType &type) const;
  };
} // namespace MeltPoolDG::LevelSet
