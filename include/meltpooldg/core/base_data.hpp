#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  struct BaseData
  {
    std::string       case_name           = "not_initialized";
    std::string       problem_name        = "not_initialized";
    unsigned int      dimension           = 2;
    unsigned int      global_refinements  = 1;
    bool              do_print_parameters = true;
    unsigned int      verbosity_level     = 1;
    FiniteElementData fe;

    BaseData();

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const unsigned int ls_n_subdivisions) const;
  };
} // namespace MeltPoolDG
