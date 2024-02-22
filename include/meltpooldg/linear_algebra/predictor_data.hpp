#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

namespace MeltPoolDG
{
  // choose the particular predictor type for the nonlinear/linear solver
  BETTER_ENUM(PredictorType,
              char,
              // no predictor specified; use old value as initial guess
              none,
              // calculate the predictor by a linear combination from the two old solution vectors
              linear_extrapolation,
              // least squares projection (WIP)
              least_squares_projection)


  template <typename number = double>
  struct PredictorData
  {
  public:
    PredictorType type                   = PredictorType::none;
    unsigned int  n_old_solution_vectors = 2;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    post();
  };
} // namespace MeltPoolDG