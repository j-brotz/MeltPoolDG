#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/hyperbolic_pde_tools/fe_subcell_evaluation.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

namespace MeltPoolDG::HyperbolicPDETools
{
  template <int dim, int n_components, typename number>
  FESubcellEvaluation<dim, n_components, number>::FESubcellEvaluation(
    const dealii::MatrixFree<dim, number> &matrix_free,
    const unsigned int                     dof_no,
    const unsigned int                     quad_no)
    : matrix_free_context(matrix_free, dof_no, quad_no)
    , fe_cell_integrator(matrix_free, dof_no, quad_no)
    , fe_inner_face_integrator(matrix_free, true, dof_no, quad_no)
    , fe_outer_face_integrator(matrix_free, false, dof_no, quad_no)
    , n_subcells_1d(matrix_free.get_quadrature(quad_no).get_tensor_basis()[0].size())
    , n_padded_subcells_1d(n_subcells_1d + 2)
    , n_subcells(std::pow(n_subcells_1d, dim))
  {
    subcell_values.resize(std::pow(n_padded_subcells_1d, dim));
    submitted_subcell_values.resize(n_subcells);
  }


  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::reinit(const unsigned int cell_batch_index_in)
  {
    cell_batch_index = cell_batch_index_in;
    fe_cell_integrator.reinit(cell_batch_index);
  }


  template <int dim, int n_components, typename number>
  dealii::std_cxx20::ranges::iota_view<unsigned int, unsigned int>
  FESubcellEvaluation<dim, n_components, number>::subcell_indices() const
  {
    return fe_cell_integrator.quadrature_point_indices();
  }


  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::gather_evaluate(
    const dealii::LinearAlgebra::distributed::Vector<number> &input_vector,
    std::function<value_type(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                             dealii::types::boundary_id,
                             const value_type &)>             get_boundary_value)
  {
    fe_cell_integrator.gather_evaluate(input_vector, dealii::EvaluationFlags::values);

    for (unsigned int q : fe_cell_integrator.quadrature_point_indices())
      {
        subcell_values[subcell_index_to_padded_index(q)] = fe_cell_integrator.get_value(q);
      }

    for (unsigned int face_no = 0; face_no < 2 * dim; ++face_no)
      {
        const auto face_q_index_to_padded_index = [this, face_no](unsigned int face_q_index) {
          const unsigned int direction = face_no / 2;
          const unsigned int side      = face_no % 2;

          unsigned int padded_index = 0;
          unsigned int stride       = 1;
          for (unsigned int d = 0; d < dim; ++d)
            {
              const unsigned int coordinate_d = (d == direction) ?
                                                  ((side == 0) ? 0 : n_subcells_1d + 1) :
                                                  (face_q_index % n_subcells_1d + 1);
              if (d != direction)
                face_q_index /= n_subcells_1d;

              padded_index += coordinate_d * stride;
              stride *= n_padded_subcells_1d;
            }
          return padded_index;
        };

        const auto boundary_ids =
          matrix_free_context.mf.get_faces_by_cells_boundary_id(cell_batch_index, face_no);
        const bool is_at_boundary = boundary_ids[0] != dealii::numbers::internal_face_boundary_id;

        fe_inner_face_integrator.reinit(cell_batch_index, face_no);
        if (is_at_boundary)
          {
            Assert(
              get_boundary_value,
              dealii::ExcMessage(
                "FESubcellEvaluation requires a boundary value function to be provided if working on a cell batch that has a face at the boundary."));

            fe_inner_face_integrator.gather_evaluate(input_vector, dealii::EvaluationFlags::values);
            for (unsigned int q : fe_inner_face_integrator.quadrature_point_indices())
              {
                subcell_values[face_q_index_to_padded_index(q)] =
                  get_boundary_value(fe_inner_face_integrator.quadrature_point(q),
                                     boundary_ids[0],
                                     fe_inner_face_integrator.get_value(q));
              }
          }
        else
          {
            fe_outer_face_integrator.reinit(cell_batch_index, face_no);
            fe_outer_face_integrator.gather_evaluate(input_vector, dealii::EvaluationFlags::values);
            for (unsigned int q : fe_inner_face_integrator.quadrature_point_indices())
              {
                subcell_values[face_q_index_to_padded_index(q)] =
                  fe_outer_face_integrator.get_value(q);
              }
          }
      }

    // Copy the subcell values to the submitted_subcell_values vector, which will be used for
    // setting the DoF values later. This is done in the case the user only writes a subset of the
    // subcell values, and we want to ensure that the submitted_subcell_values vector contains the
    // most up-to-date values for all subcells before writing them to the DoF vector.
    for (unsigned int subcell_index = 0; subcell_index < n_subcells; ++subcell_index)
      {
        submitted_subcell_values[subcell_index] =
          subcell_values[subcell_index_to_padded_index(subcell_index)];
      }
  }


  template <int dim, int n_components, typename number>
  dealii::VectorizedArray<number>
  FESubcellEvaluation<dim, n_components, number>::subcell_size(
    const unsigned int subcell_index) const
  {
    AssertIndexRange(subcell_index, n_subcells);
    return fe_cell_integrator.JxW(subcell_index);
  }


  template <int dim, int n_components, typename number>
  dealii::VectorizedArray<number>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_size(
    const unsigned int subcell_index,
    const unsigned int face_no) const
  {
    AssertIndexRange(subcell_index, n_subcells);
    AssertIndexRange(face_no, 2 * dim);

    const unsigned int direction = face_no / 2;

    const dealii::Quadrature<1> quadrature_1d =
      matrix_free_context.mf.get_quadrature(matrix_free_context.quad_idx).get_tensor_basis()[0];

    dealii::VectorizedArray<number> reference_tangential_extent(1.);
    unsigned int                    idx = subcell_index;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const unsigned int coordinate_d = idx % n_subcells_1d;
        idx /= n_subcells_1d;
        if (d != direction)
          reference_tangential_extent *= quadrature_1d.weight(coordinate_d);
      }

    const auto inverse_jacobian     = fe_cell_integrator.inverse_jacobian(0);
    const auto jacobian_determinant = 1. / dealii::determinant(inverse_jacobian);

    return jacobian_determinant * inverse_jacobian[direction][direction] *
           reference_tangential_extent;
  }

  template <int dim, int n_components, typename number>
  dealii::Tensor<1, dim, dealii::VectorizedArray<number>>
  FESubcellEvaluation<dim, n_components, number>::subcell_face_normal(
    const unsigned int subcell_index,
    const unsigned int face_no)
  {
    (void)subcell_index; // Currently not used but already present for possible extensions in the
                         // future.
    AssertIndexRange(subcell_index, n_subcells);
    AssertIndexRange(face_no, 2 * dim);

    const unsigned int direction = face_no / 2;
    const unsigned int side      = face_no % 2;

    dealii::Tensor<1, dim, dealii::VectorizedArray<number>> normal;
    normal[direction] = (side == 0) ? -1. : 1;
    return normal;
  }


  template <int dim, int n_components, typename number>
  typename FESubcellEvaluation<dim, n_components, number>::value_type
  FESubcellEvaluation<dim, n_components, number>::get_subcell_value(
    const unsigned int subcell_index) const
  {
    AssertIndexRange(subcell_index, n_subcells);
    return subcell_values[subcell_index_to_padded_index(subcell_index)];
  }


  template <int dim, int n_components, typename number>
  typename FESubcellEvaluation<dim, n_components, number>::value_type
  FESubcellEvaluation<dim, n_components, number>::get_subcell_neighbor_value(
    const unsigned int subcell_index,
    const unsigned int face_no) const
  {
    AssertIndexRange(subcell_index, n_subcells);
    AssertIndexRange(face_no, 2 * dim);

    const unsigned int direction = face_no / 2;
    const unsigned int side      = face_no % 2;

    return subcell_values[subcell_index_to_padded_index(subcell_index) +
                          ((side == 0) ? -1 : 1) *
                            static_cast<unsigned int>(std::pow(n_padded_subcells_1d, direction))];
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::submit_subcell_value(
    const unsigned int subcell_index,
    const value_type  &value)
  {
    AssertIndexRange(subcell_index, submitted_subcell_values.size());
    submitted_subcell_values[subcell_index] = value;
  }

  template <int dim, int n_components, typename number>
  void
  FESubcellEvaluation<dim, n_components, number>::set_dof_values(
    dealii::LinearAlgebra::distributed::Vector<number> &dst_vector)
  {
    dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number> inverse(
      fe_cell_integrator);

    std::vector<dealii::VectorizedArray<number>> subcell_values_dof_vector(
      n_components * fe_cell_integrator.quadrature_point_indices().size());

    for (unsigned int subcell : fe_cell_integrator.quadrature_point_indices())
      {
        if constexpr (n_components == 1)
          subcell_values_dof_vector[subcell] = submitted_subcell_values[subcell];
        else
          for (unsigned int c = 0; c < n_components; ++c)
            subcell_values_dof_vector[c * fe_cell_integrator.quadrature_point_indices().size() +
                                      subcell] = submitted_subcell_values[subcell][c];
      }
    inverse.transform_from_q_points_to_basis(n_components,
                                             subcell_values_dof_vector.data(),
                                             fe_cell_integrator.begin_dof_values());
    fe_cell_integrator.set_dof_values(dst_vector);
  }

  template <int dim, int n_components, typename number>
  bool
  FESubcellEvaluation<dim, n_components, number>::is_subcell_face_at_cell_boundary(
    const unsigned int subcell_index,
    const unsigned int face_no) const
  {
    AssertIndexRange(subcell_index, n_subcells);
    AssertIndexRange(face_no, 2 * dim);

    const unsigned int direction = face_no / 2;
    const unsigned int side      = face_no % 2;

    unsigned int idx = subcell_index;
    for (unsigned int d = 0; d < direction; ++d)
      idx /= n_subcells_1d;

    const unsigned int i_direction = idx % n_subcells_1d;

    return (side == 0) ? (i_direction == 0) : (i_direction == n_subcells_1d - 1);
  }


  template <int dim, int n_components, typename number>
  unsigned int
  FESubcellEvaluation<dim, n_components, number>::subcell_index_to_padded_index(
    unsigned int subcell_index) const
  {
    unsigned int padded_index = 0;
    unsigned int stride       = 1;
    for (unsigned int d = 0; d < dim; ++d)
      {
        const unsigned int coordinate_d = subcell_index % n_subcells_1d;
        subcell_index /= n_subcells_1d;

        padded_index += (coordinate_d + 1) * stride;
        stride *= n_padded_subcells_1d;
      }
    return padded_index;
  }
} // namespace MeltPoolDG::HyperbolicPDETools
