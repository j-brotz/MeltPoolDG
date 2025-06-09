#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  // choose the particular predictor type for the nonlinear/linear solver
  BETTER_ENUM(PredictorType,
              char,
              // use old value as initial guess
              none,
              // use zeros as initial guess
              zero,
              // calculate the predictor by a linear combination from the two old solution vectors
              linear_extrapolation,
              // least squares projection (WIP)
              least_squares_projection)


  struct PredictorData
  {
    PredictorType type = PredictorType::none;

    unsigned int n_old_solution_vectors =
      2; // only relevant for least_squares_projection, otherwise set appropriately

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post();

    void
    check_input_parameters() const;
  };
} // namespace MeltPoolDG