#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG::Profiling
{
  BETTER_ENUM(TimeType, char, real, simulation)

  template <typename number>
  struct ProfilingData
  {
    // Mutable because check input parameters may set enable to true if any of the individual
    // profiling options is enabled but the applications require check_input_parameters to be const.
    mutable bool enable = false;

    bool enable_wall_time_statistics = false;
    bool enable_iteration_statistics = false;
    bool enable_dof_statistics       = false;
    bool enable_cell_statistics      = false;

    number   write_time_step_size = 10.0;
    TimeType time_type            = TimeType::real;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const number time_step_size) const;
  };

} // namespace MeltPoolDG::Profiling