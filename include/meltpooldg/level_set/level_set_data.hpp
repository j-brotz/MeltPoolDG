#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/level_set/nearest_point_data.hpp>

#include <limits>
#include <string>

namespace MeltPoolDG::LevelSet
{
  template <typename number = double>
  struct LevelSetData
  {
    bool                     do_reinitialization     = true;
    int                      n_initial_reinit_steps  = -1.0;
    std::string              time_integration_scheme = "crank_nicolson";
    bool                     do_curvature_correction = false;
    int                      n_subdivisions          = 1;
    bool                     do_localized_heaviside  = true;
    std::string              implementation          = "meltpooldg";
    number                   reinit_time_step_size   = -1.;
    number                   tol_reinit              = std::numeric_limits<number>::min();
    NearestPointData<number> nearest_point;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const unsigned int reinit_max_n_steps);

    void
    check_input_parameters(const unsigned int base_degree) const;
  };
} // namespace MeltPoolDG::LevelSet