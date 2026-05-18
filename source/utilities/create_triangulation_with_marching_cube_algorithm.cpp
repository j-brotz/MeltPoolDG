#include <meltpooldg/utilities/create_triangulation_with_marching_cube_algorithm.hpp>
//
#include <deal.II/base/point.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_tools.h>

#include <vector>

namespace MeltPoolDG::GridGenerator
{
  template <int dim, typename number>
  void
  create_triangulation_with_marching_cube_algorithm(
    dealii::Triangulation<dim - 1, dim>                      &tria,
    const dealii::Mapping<dim>                               &mapping,
    const dealii::DoFHandler<dim>                            &background_dof_handler,
    const dealii::LinearAlgebra::distributed::Vector<number> &ls_vector,
    const number                                              iso_level,
    const unsigned int                                        n_subdivisions,
    const number                                              tolerance)
  {
    std::vector<dealii::Point<dim>>        vertices;
    std::vector<dealii::CellData<dim - 1>> cells;
    dealii::SubCellData                    subcelldata;

    const dealii::GridTools::
      MarchingCubeAlgorithm<dim, dealii::LinearAlgebra::distributed::Vector<number>>
        mc(mapping, background_dof_handler.get_fe(), n_subdivisions, tolerance);

    const bool vector_is_ghosted = ls_vector.has_ghost_elements();

    if (vector_is_ghosted == false)
      ls_vector.update_ghost_values();

    mc.process(background_dof_handler, ls_vector, iso_level, vertices, cells);

    if (vector_is_ghosted == false)
      ls_vector.zero_out_ghost_values();

    std::vector<unsigned int> considered_vertices;

    // note: the following operation does not work for simplex meshes yet
    // GridTools::delete_duplicated_vertices (vertices, cells, subcelldata,
    // considered_vertices);

    if (vertices.size() > 0)
      tria.create_triangulation(vertices, cells, subcelldata);
  }

  template void
  create_triangulation_with_marching_cube_algorithm(
    dealii::Triangulation<1, 2> &,
    const dealii::Mapping<2> &,
    const dealii::DoFHandler<2> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const double,
    const unsigned int,
    const double);
  template void
  create_triangulation_with_marching_cube_algorithm(
    dealii::Triangulation<2, 3> &,
    const dealii::Mapping<3> &,
    const dealii::DoFHandler<3> &,
    const dealii::LinearAlgebra::distributed::Vector<double> &,
    const double,
    const unsigned int,
    const double);
} // namespace MeltPoolDG::GridGenerator
