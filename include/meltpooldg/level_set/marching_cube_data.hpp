#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  struct MarchingCubeData
  {
    number       tolerance      = 1e-10;
    unsigned int n_subdivisions = 3;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };
} // namespace MeltPoolDG::LevelSet
