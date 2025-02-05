#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/matrix_free/matrix_free.h>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  /**
   * This function computes the local values of the internal penalty parameter used in the viscous
   * numerical flux.
   *
   * @param array_penalty_parameter Array in which the values of the penalty parameter are stored.
   * @param matrix_free Matrix-free object providing the required geometrical data.
   * @param domain_representation_type Type of geometrical domain representation (cut or fitted).
   * @param dof_index Index of the relevant dof handler in the matrix-free object.
   * @param scaling_factor Additional scaling factor to scale the penalty parameter.
   */
  template <int dim, typename Number>
  void
  calculate_penalty_parameter(AlignedVector<VectorizedArray<Number>> &array_penalty_parameter,
                              const MatrixFree<dim, Number>          &matrix_free,
                              const std::string                      &domain_representation_type,
                              const unsigned int                      dof_index      = 0,
                              const double                            scaling_factor = 1.0)
  {
    const unsigned int n_cells = matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_cells);

    Mapping<dim> const       &mapping = *matrix_free.get_mapping_info().mapping;
    FiniteElement<dim> const &fe      = matrix_free.get_dof_handler(dof_index).get_fe();
    unsigned int const        degree  = fe.degree;

    // use penalty factor for hypercube elements according to K. Hillewaert, Development of the
    // discontinuous Galerkin method for high-resolution, large scale CFD and acoustics in
    // industrial geometries, PhD thesis, Univ. de Louvain, 2013.
    const double fac = scaling_factor * (degree + 1.0) * (degree + 1.0);

    auto const reference_cells =
      matrix_free.get_dof_handler(dof_index).get_triangulation().get_reference_cells();
    AssertThrow(reference_cells.size() == 1, ExcMessage("No mixed meshes allowed."));

    auto const quadrature = reference_cells[0].template get_gauss_type_quadrature<dim>(degree + 1);
    FEValues<dim> fe_values(mapping, fe, quadrature, update_JxW_values);

    auto const face_quadrature =
      reference_cells[0].face_reference_cell(0).template get_gauss_type_quadrature<dim - 1>(degree +
                                                                                            1);
    FEFaceValues<dim> fe_face_values(mapping, fe, face_quadrature, update_JxW_values);

    if (domain_representation_type == "fitted")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);
                fe_values.reinit(cell);

                // calculate cell volume
                Number volume = 0;
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  {
                    volume += fe_values.JxW(q);
                  }

                // calculate surface area
                Number surface_area = 0;
                for (unsigned int const f : cell->face_indices())
                  {
                    fe_face_values.reinit(cell, f);
                    Number const factor =
                      (cell->at_boundary(f) and not(cell->has_periodic_neighbor(f))) ? 1. : 0.5;
                    for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                      {
                        surface_area += fe_face_values.JxW(q) * factor;
                      }
                  }

                array_penalty_parameter[i][v] = surface_area / volume * fac;
              }
          }
      }
    else if (domain_representation_type == "cut")
      {
        for (unsigned int i = 0; i < n_cells; ++i)
          {
            for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(i); ++v)
              {
                typename DoFHandler<dim>::cell_iterator cell =
                  matrix_free.get_cell_iterator(i, v, dof_index);

                // simplified computation for hypercube elements
                array_penalty_parameter[i][v] = fac / cell->minimum_vertex_distance();
              }
          }
      }
    else
      AssertThrow(false,
                  dealii::ExcMessage("The domain representation type '" +
                                     domain_representation_type + "' is not supported."));
  }
} // namespace MeltPoolDG::Flow
