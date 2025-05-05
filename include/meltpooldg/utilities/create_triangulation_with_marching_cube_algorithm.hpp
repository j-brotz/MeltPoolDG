#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>


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
    const unsigned int                                        n_subdivisions = 1,
    const number                                              tolerance      = 1e-10);
} // namespace MeltPoolDG::GridGenerator