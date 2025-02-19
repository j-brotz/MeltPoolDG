#include <deal.II/base/patterns.h>

#include <meltpooldg/level_set/marching_cube_data.hpp>

#include <limits>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  void
  MarchingCubeData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("marching cube");
    {
      prm.add_parameter("n subdivisions",
                        n_subdivisions,
                        "Specify the number of subdivisions to create a quadrature rule with "
                        "n_subdivisions+1 equally-positioned quadrature points.",
                        dealii::Patterns::Integer(1, std::numeric_limits<unsigned int>::max()));
      prm.add_parameter("tol",
                        tolerance,
                        "Absolute tolerance specifying the minimum distance between a vertex and "
                        "the cut point so that a line is considered cut.",
                        dealii::Patterns::Double(0, std::numeric_limits<number>::max()));
    }
    prm.leave_subsection();
  }

  template struct MarchingCubeData<double>;
} // namespace MeltPoolDG::LevelSet
