#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

#include <string>

namespace MeltPoolDG::Restart
{
  BETTER_ENUM(TimeType, char, real, simulation)

  template <typename number = double>
  struct RestartData
  {
    int         load                 = -1;
    int         save                 = -1;
    std::string directory            = "";
    std::string prefix               = "restart";
    double      write_time_step_size = 0.0;
    TimeType    time_type            = TimeType::real;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const std::string &output_directory);

    void
    check_input_parameters(const number time_step_size) const;
  };
} // namespace MeltPoolDG::Restart