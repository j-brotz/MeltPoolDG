/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, May 2022
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/parameter_handler.h>

#include <string>

namespace MeltPoolDG
{
  template <typename number>
  struct TimeSteppingData
  {
    number       start_time              = 0.0;
    number       end_time                = 1.0;
    number       time_step_size          = 0.01;
    unsigned int max_n_steps             = 10000000;
    std::string  time_step_size_function = "0.0*t";

    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  template <typename number>
  struct TimeStepLimitData
  {
    bool   enable       = false;
    number scale_factor = 1.0;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG
