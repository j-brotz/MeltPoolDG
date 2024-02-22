#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/nonlinear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>

namespace MeltPoolDG::Heat
{
  template <typename number = double>
  struct HeatData
  {
    HeatData();

    int                                                   degree                   = -1;
    int                                                   n_subdivisions           = 1;
    int                                                   n_q_points_1d            = -1;
    number                                                emissivity               = 0.0;
    number                                                convection_coefficient   = 0.0;
    number                                                temperature_infinity     = 0.0;
    bool                                                  enable_time_dependent_bc = false;
    NonlinearSolverData<number>                           nlsolve;
    LinearSolverData<number>                              linear_solver;
    LevelSet::DeltaApproximationPhaseWeightedData<number> delta_approximation_phase_weighted;
    bool                  use_volume_specific_thermal_capacity_for_phase_interpolation = false;
    PredictorData<number> predictor;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const unsigned int          base_degree,
         const unsigned int          base_verbosity_level,
         const MaterialData<number> &material);

    void
    check_input_parameters(const bool base_do_simplex, const int ls_n_subdivisions) const;
  };
} // namespace MeltPoolDG::Heat