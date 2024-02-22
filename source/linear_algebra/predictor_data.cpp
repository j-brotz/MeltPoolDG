#include <meltpooldg/linear_algebra/predictor_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  PredictorData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("predictor");
    {
      prm.add_parameter("type", type, "Choose a predictor type.");
      prm.add_parameter("n old solutions",
                        n_old_solution_vectors,
                        "Choose the number of old solution vectors considered."
                        "This parameter is only relevant for least squares projection.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  PredictorData<number>::post()
  {
    if (type == PredictorType::none)
      n_old_solution_vectors = 1;
    else if (type == PredictorType::linear_extrapolation)
      n_old_solution_vectors = 2;
  }

  template struct PredictorData<double>;
} // namespace MeltPoolDG