#include <meltpooldg/linear_algebra/predictor_data.hpp>
//
#include <deal.II/base/exceptions.h>

namespace MeltPoolDG
{
  void
  PredictorData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("predictor");
    {
      prm.add_parameter(
        "type",
        type,
        "Choose a predictor type: "
        "none: use old value as initial guess; "
        "zero: se zeros as initial guess; "
        "linear_extrapolation: calculate the predictor by a linear combination from the two old solution vectors; "
        "least_squares_projection: least squares projection (WIP)");

      prm.add_parameter("n old solutions",
                        n_old_solution_vectors,
                        "Choose the number of old solution vectors considered."
                        "This parameter is only relevant for least squares projection."
                        "For all other predictors, this parameter will be set appropriately.");
    }
    prm.leave_subsection();
  }

  void
  PredictorData::post()
  {
    switch (type)
      {
        case PredictorType::none:
          case PredictorType::zero: {
            n_old_solution_vectors = 1;
            break;
          }
          case PredictorType::linear_extrapolation: {
            n_old_solution_vectors = 2;
          }
        case PredictorType::least_squares_projection:
        default:
          break;
      }
  }

  void
  PredictorData::check_input_parameters() const
  {
    AssertThrow(n_old_solution_vectors > 0,
                dealii::ExcMessage("The solution history cannot have a length of zero."));
  }
} // namespace MeltPoolDG
