#include <meltpooldg/level_set/nearest_point_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  void
  NearestPointData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("nearest point");
    {
      prm.add_parameter(
        "max iter",
        max_iter,
        "Maximum number of corrections of the point projection towards the interface.");
      prm.add_parameter("rel tol",
                        rel_tol,
                        "Relative tolerance to be achieved within the projection.");
      prm.add_parameter("narrow band threshold",
                        narrow_band_threshold,
                        "Maximum value of the level set for defining narrow band where "
                        "CPP is performed.");
      prm.add_parameter("type",
                        type,
                        "Choose the type for calculating the nearest point to the interface.");
      prm.add_parameter("verbosity level", verbosity_level, "Set the verbosity level.");
    }
    prm.leave_subsection();
  }

  template struct NearestPointData<double>;
} // namespace MeltPoolDG::LevelSet