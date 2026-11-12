#include <meltpooldg/utilities/limiter_data.hpp>

namespace MeltPoolDG::Utilities
{
  template <typename number>
  void
  LimiterData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("limiter");
    {
      prm.add_parameter(
        "output troubled cells",
        output_troubled_cells,
        "Whether to output the troubled cell markers for the applied marking strategies.");
      prm.add_parameter("type", type, "The type of limiter to apply.");
      prm.add_parameter("limiter function type",
                        limiter_function_type,
                        "The type of limiter function to use. It limits the slopes of the slope "
                        "limiter and of the second order subcell reconstruction of the finite "
                        "volume subcell limiter.");
      prm.add_parameter(
        "subcell reconstruction order",
        subcell_reconstruction_order,
        "The order of the reconstruction of the subcell face values of the finite volume subcell "
        "limiter: 1 for piecewise constant subcell values, 2 for a piecewise linear reconstruction "
        "with slopes limited by the limiter function type.",
        dealii::Patterns::Integer(1, 2));
      prm.add_parameter(
        "troubled cell marking strategy",
        troubled_cell_marking_strategy,
        "The troubled cell marking strategy used to identify cells that require limiting.");
    }
    prm.leave_subsection();
  }

  template struct LimiterData<double>;
} // namespace MeltPoolDG::Utilities
