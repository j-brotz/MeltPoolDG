#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/interface/finite_element_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>

namespace MeltPoolDG::Heat
{
  template <typename number = double>
  struct HeatData
  {
    HeatData();

    bool enable_time_dependent_bc                                     = false;
    bool use_volume_specific_thermal_capacity_for_phase_interpolation = false;

    struct RadiationBC
    {
      number emissivity           = 0.0;
      number temperature_infinity = 0.0;
    } radiation;

    struct ConvectionBC
    {
      number convection_coefficient = 0.0;
      number temperature_infinity   = 0.0;
    } convection;

    NonlinearSolverData<number> nlsolve;
    LinearSolverData<number>    linear_solver;
    PredictorData<number>       predictor;

    FiniteElementData fe;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const FiniteElementData &base_fe_data, const unsigned int base_verbosity_level);

    void
    check_input_parameters(const FiniteElementData &base_fe_data) const;
  };
} // namespace MeltPoolDG::Heat
