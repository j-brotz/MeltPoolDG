#include <meltpooldg/utilities/triangulation_type.hpp>
//
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

namespace MeltPoolDG
{
  template <int dim, int spacedim>
  TriangulationType
  get_triangulation_type(const dealii::Triangulation<dim, spacedim> &tria)
  {
    if (dynamic_cast<dealii::parallel::shared::Triangulation<dim, spacedim> *>(
          const_cast<dealii::Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::shared;
    else if (dynamic_cast<dealii::parallel::distributed::Triangulation<dim, spacedim> *>(
               const_cast<dealii::Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::parallel_distributed;
    else if (dynamic_cast<dealii::parallel::fullydistributed::Triangulation<dim, spacedim> *>(
               const_cast<dealii::Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::parallel_fullydistributed;
    else
      return TriangulationType::serial;
  }

  template TriangulationType
  get_triangulation_type(const dealii::Triangulation<1, 1> &);
  template TriangulationType
  get_triangulation_type(const dealii::Triangulation<2, 2> &);
  template TriangulationType
  get_triangulation_type(const dealii::Triangulation<3, 3> &);

} // namespace MeltPoolDG