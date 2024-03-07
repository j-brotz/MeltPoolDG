#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/linear_algebra/predictor_data.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <string>

namespace MeltPoolDG::LevelSet
{
  BETTER_ENUM(InterfaceThicknessParameterType,
              char,
              // epsilon = value * cell size
              // interface thickness = value * 6 * cell size
              proportional_to_cell_size,
              // epsilon = value
              // interface thickness = value * 6
              absolute_value,
              // epsilon = value * cell size / 6
              // interface thickness = value * cell size
              number_of_cells_across_interface)

  template <typename number = double>
  struct ReinitializationData
  {
    ReinitializationData();

    bool         enable          = true;
    unsigned int max_n_steps     = 5;
    int          n_initial_steps = -1;
    number       tolerance       = std::numeric_limits<number>::min();

    struct InterfaceThickness
    {
      InterfaceThicknessParameterType type =
        InterfaceThicknessParameterType::proportional_to_cell_size;
      number value = 0.5;
    } interface_thickness_parameter;

    std::string              modeltype      = "olsson2007";
    std::string              implementation = "meltpooldg";
    PredictorData<number>    predictor;
    LinearSolverData<number> linear_solver;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const bool normal_vec_do_matrix_free) const;

    void
    post();

    template <typename number2>
    number2
    compute_interface_thickness_parameter_epsilon(const number2 cell_size) const
    {
      switch (interface_thickness_parameter.type)
        {
          case InterfaceThicknessParameterType::proportional_to_cell_size:
            return cell_size * interface_thickness_parameter.value;
          case InterfaceThicknessParameterType::absolute_value:
            return interface_thickness_parameter.value;
          case InterfaceThicknessParameterType::number_of_cells_across_interface:
            return interface_thickness_parameter.value * cell_size / 6.;
          default:
            AssertThrow(false, ExcNotImplemented());
            return 0.0;
        }
    }
  };
} // namespace MeltPoolDG::LevelSet
