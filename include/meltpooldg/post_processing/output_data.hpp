#pragma once

#include <deal.II/base/parameter_handler.h>

#include <string>
#include <vector>

namespace MeltPoolDG
{
  struct ParaviewData
  {
    bool                     enable                   = true;
    std::string              filename                 = "solution";
    int                      n_digits_timestep        = 4;
    std::vector<std::string> output_variables         = {"all"};
    bool                     print_boundary_id        = false;
    bool                     output_subdomains        = false;
    bool                     output_material_id       = false;
    bool                     write_higher_order_cells = true;
    int                      n_groups                 = 1;
    int                      n_patches                = 0;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  template <typename number = double>
  struct OutputData
  {
    bool         do_output            = false;
    std::string  directory            = "./";
    int          write_frequency      = 1;
    number       write_time_step_size = 0.0;
    ParaviewData paraview;
    bool         do_user_defined_postprocessing = false;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post(const number time_step_size, const std::string &parameter_filename);
  };
} // namespace MeltPoolDG