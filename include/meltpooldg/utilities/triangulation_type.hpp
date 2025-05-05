#pragma once

#include <deal.II/grid/tria.h>

namespace MeltPoolDG
{
  enum TriangulationType
  {
    shared,
    parallel_distributed,
    parallel_fullydistributed,
    serial
  };

  template <int dim, int spacedim = dim>
  TriangulationType
  get_triangulation_type(const dealii::Triangulation<dim, spacedim> &tria);
} // namespace MeltPoolDG