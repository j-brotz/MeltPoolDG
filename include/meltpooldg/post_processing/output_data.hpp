#pragma once

#include <deal.II/base/parameter_handler.h>

#include <string>
#include <vector>

namespace MeltPoolDG
{
  template <typename number = double>
  struct OutputData
  {
    bool                     do_output                      = false;
    bool                     do_user_defined_postprocessing = false;
    std::string              filename                       = "solution";
    std::string              directory                      = "./";
    int                      write_frequency                = 1;
    double                   write_time_step_size           = 0.0;
    bool                     print_boundary_id              = false;
    bool                     output_subdomains              = false;
    bool                     output_material_id             = false;
    int                      n_digits_timestep              = 4;
    int                      n_groups                       = 1;
    int                      n_patches                      = 0;
    bool                     write_higher_order_cells       = true;
    std::vector<std::string> output_variables               = {"all"};

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const number time_step_size, const std::string &parameter_filename);
  };
} // namespace MeltPoolDG