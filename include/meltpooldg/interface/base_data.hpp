#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

#include <string>

namespace MeltPoolDG
{
  BETTER_ENUM(ProblemType,
              char,
              advection_diffusion,
              reinitialization,
              level_set,
              melt_pool,
              level_set_with_evaporation,
              heat_transfer,
              radiative_transport,
              none)

  template <typename number = double>
  struct BaseData
  {
    std::string  application_name    = "none";
    ProblemType  problem_name        = ProblemType::advection_diffusion;
    unsigned int dimension           = 2;
    unsigned int global_refinements  = 1;
    unsigned int degree              = 1;
    int          n_q_points_1d       = -1;
    bool         do_print_parameters = true;
    bool         do_simplex          = false;
    number       gravity             = 0.0;
    unsigned int verbosity_level     = 0;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post();

    void
    check_input_parameters(const unsigned int ls_n_subdivisions) const;
  };
} // namespace MeltPoolDG