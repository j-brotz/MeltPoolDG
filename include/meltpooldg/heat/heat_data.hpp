#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/cut/cut_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG::Heat
{
  BETTER_ENUM(TwoPhaseOperatorType, char, diffuse, cut)

  template <typename number = double>
  struct HeatData
  {
    HeatData();

    TwoPhaseOperatorType operator_type = TwoPhaseOperatorType::diffuse;

    struct Diffuse
    {
      bool use_volume_specific_thermal_capacity_for_phase_interpolation = false;
    } diffuse;

    struct Cut
    {
      bool two_phase = true;

      // factor theta for time integration with one-step-theta method
      // TODO move to a time integrator scheme section
      number theta = 0.5;

      bool do_explicit_symmetry_term = true;

      // cut-related stabilization parameters
      CutStabilizationData<number> stabilization;
    } cut;

    // boundary conditions

    bool enable_time_dependent_bc = false;

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

    // numerics

    NonlinearSolverData<number> nlsolve;
    LinearSolverData<number>    linear_solver;
    PredictorData<number>       predictor;

    FiniteElementData fe;

    int verbosity_level = -1;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const FiniteElementData &base_fe_data, const unsigned int base_verbosity_level);

    void
    check_input_parameters(const FiniteElementData &base_fe_data) const;
  };
} // namespace MeltPoolDG::Heat
