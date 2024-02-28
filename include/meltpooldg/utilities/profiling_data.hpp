#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG::Profiling
{
  BETTER_ENUM(TimeType, char, real, simulation)

  template <typename number = double>
  struct ProfilingData
  {
    bool     enable               = false;
    double   write_time_step_size = 10.0;
    TimeType time_type            = TimeType::real;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const unsigned int base_verbosity_level);

    void
    check_input_parameters(const number time_step_size) const;
  };

} // namespace MeltPoolDG::Profiling