#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/limiters.templates.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <utility>

namespace MeltPoolDG::Utilities
{

  template <int dim, typename cell_data_type>
  class DistributedCellData
  {
    using ActiveCellIterator = typename dealii::Triangulation<dim>::active_cell_iterator;

  public:
    explicit DistributedCellData(const dealii::Triangulation<dim> &tria_in)
      : tria(tria_in)
      , cell_data(tria.n_active_cells())
    {}

    cell_data_type &
    operator[](const dealii::types::global_cell_index active_cell_index)
    {
      AssertIndexRange(active_cell_index, tria.n_active_cells());
      return cell_data[active_cell_index];
    }

    const cell_data_type &
    operator[](const dealii::types::global_cell_index active_cell_index) const
    {
      AssertIndexRange(active_cell_index, tria.n_active_cells());
      return cell_data[active_cell_index];
    }

    cell_data_type *
    data()
    {
      return cell_data.data();
    }

    typename std::vector<cell_data_type>::iterator
    begin()
    {
      return cell_data.begin();
    }

    typename std::vector<cell_data_type>::iterator
    end()
    {
      return cell_data.end();
    }

    typename std::vector<cell_data_type>::const_iterator
    cbegin() const
    {
      return cell_data.cbegin();
    }

    typename std::vector<cell_data_type>::const_iterator
    cend() const
    {
      return cell_data.cend();
    }

    void
    update_ghost_values()
    {
      std::function<std::optional<cell_data_type>(const ActiveCellIterator &)> pack =
        [&](const ActiveCellIterator &cell) -> std::optional<cell_data_type> {
        return cell_data[cell->active_cell_index()];
      };

      std::function<void(const ActiveCellIterator &, const cell_data_type &)> unpack =
        [&](const ActiveCellIterator &cell, const cell_data_type &value) {
          cell_data[cell->active_cell_index()] = value;
        };

      dealii::GridTools::exchange_cell_data_to_ghosts(tria, pack, unpack);
    }

  private:
    const dealii::Triangulation<dim> &tria;
    std::vector<cell_data_type>       cell_data;
  };


  // TODO: This assumes a cartesian grid with NO local mesh refinement
  template <int dim, int n_components, typename number>
  class Limiter
  {
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using ValueType           = dealii::Tensor<1, n_components, VectorizedArrayType>;
    using FluxType = dealii::Tensor<1, n_components, dealii::Tensor<1, dim, VectorizedArrayType>>;

  public:
    Limiter(const LimiterData<number>             &limiter_data,
            const dealii::MatrixFree<dim, number> &matrix_free,
            const unsigned int                     dof_idx,
            const unsigned int                     quad_idx)
      : matrix_free_context(matrix_free, dof_idx, quad_idx)
      , limiter_data(limiter_data)
    {}

    void
    reinit();

    unsigned int
    mark_cells_for_limiting(
      const VectorType                                                        &solution,
      std::function<dealii::Tensor<1, n_components, number>(
        const dealii::Point<dim, number> &,
        const dealii::Tensor<1, dim, number> &,
        dealii::types::boundary_id,
        const dealii::Tensor<1, n_components, number> &)>                      get_boundary_value,
      const std::function<dealii::VectorizedArray<number>(const ValueType &)> &admissibility_check);

    void
    prepare_for_limiting(const VectorType &solution);

    unsigned int
    apply_limiting(
      const number                                                              time_step,
      const std::function<FluxType(const ValueType &w_m, const ValueType &w_p)> numerical_flux,
      std::function<dealii::Tensor<1, n_components, number>(
        const dealii::Point<dim, number> &,
        const dealii::Tensor<1, dim, number> &,
        dealii::types::boundary_id,
        const dealii::Tensor<1, n_components, number> &)>                       get_boundary_value,
      VectorType                                                               &limited_solution,
      const VectorType                                                         &solution,
      const std::function<dealii::VectorizedArray<number>(const ValueType &)> &admissibility_check =
        nullptr);

    /**
     * This function attaches relevant limiter data to the provided GenericDataOut object for output
     * generation. The data attached include information about which cells have been limited during
     * the last time the limiter has been applied. A value of zero indicates that the cell has not
     * been limited, while a value of one indicates that the cell has been limited. The data is
     * attached as an element-wise data vector with the name "limited_cells".
     *
     * @param data_out The GenericDataOut object to which the limiter data will be attached.
     *
     * @note The function must be called for each output step as it internally prepares the data to
     * be attached based on the current state of the limiter.
     */
    void
    attach_to_data_out(GenericDataOut<dim, number> &data_out) const;

  private:
    dealii::AlignedVector<dealii::VectorizedArray<number>> troubled_cells;

    mutable dealii::Vector<number> troubled_cells_output;

    VectorType previous_time_solution;

    MatrixFreeContext<dim, number> matrix_free_context;

    const LimiterData<number> limiter_data;

    /**
     * Cartesian-indexed access to the finite volume subcell average values on a cell, including the
     * neighboring subcell values needed to compute fluxes across the cell boundary.
     *
     * Each cell is subdivided into subcells centered at its quadrature points, one subcell per
     * quadrature point (see get_cartesian_from_lexicographic_q_index()). This struct stores those
     * subcell values, plus one layer of neighboring subcell values on each side, and lets you
     * address all of them via a per-direction cartesian index instead of a flat offset into the
     * underlying vector.
     *
     * For example. let's consider a cell with n_q_points_1d == 2 subcells per direction. Then the
     * 2D layout looks like this (index pair is (i, j), i.e. (x-index, y-index)):
     *
     * @verbatim
     *                     i = -1     i = 0      i = 1      i = 2
     *                   +----------+----------+----------+----------+
     *   j = 2  (north)  |          | (0, 2) N | (1, 2) N |          |
     *                   +----------+----------+----------+----------+
     *   j = 1           | (-1,1) W | (0, 1)   | (1, 1)   | (2, 1) E |
     *                   +----------+----------+----------+----------+
     *   j = 0           | (-1,0) W | (0, 0)   | (1, 0)   | (2, 0) E |
     *                   +----------+----------+----------+----------+
     *   j = -1 (south)  |          | (0,-1) S | (1,-1) S |          |
     *                   +----------+----------+----------+----------+
     * @endverbatim
     *
     * Indices in `[0, n_q_points_1d)` (unmarked cells above) address subcells that belong to this
     * cell. An index of `-1` or `n_q_points_1d` in a given direction (marked N/S/E/W) addresses the
     * adjacent subcell of the neighboring cell across that face instead: `-1` for the neighbor in
     * the negative direction, `n_q_points_1d` for the neighbor in the positive direction. Diagonal
     * (corner) neighbors are not represented, since only face-adjacent values are needed for the
     * finite volume flux computation. The same methodology extends to 3D, where the subcell values
     * are indexed by a triplet of indices (i, j, k) instead of a pair.
     */
    struct CartesianIndexedSubcellValues
    {
      /**
       * Assuming that both have been reinited on the the cell of interest
       */
      CartesianIndexedSubcellValues(
        std::function<dealii::Tensor<1, n_components, number>(
          const dealii::Point<dim, number> &,
          const dealii::Tensor<1, dim, number> &,
          dealii::types::boundary_id,
          const dealii::Tensor<1, n_components, number> &)> get_boundary_value,
        const FECellIntegrator<dim, n_components, number>  &cell_evaluator,
        FEFaceIntegrator<dim, n_components, number>        &outer_face_evaluator,
        FEFaceIntegrator<dim, n_components, number>        &inner_face_evaluator,
        const VectorType                                   &solution,
        const unsigned int                                  cell_batch_index,
        const unsigned int                                  n_q_points_1d_in)
        : n_q_points_1d(n_q_points_1d_in)
      {
        subcell_values.reserve(cell_evaluator.n_q_points +
                               2 * dim * dealii::Utilities::fixed_power<dim - 1>(n_q_points_1d));

        // interior subcell values
        for (const unsigned int q : cell_evaluator.quadrature_point_indices())
          {
            subcell_values.push_back(cell_evaluator.get_value(q));
          }

        // out subcell values (face-adjacent neighbors)
        // TODO: For now we assume that no local mesh refinement is present
        for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          {
            outer_face_evaluator.reinit(cell_batch_index, face);
            inner_face_evaluator.reinit(cell_batch_index, face);

            // Note: FEFaceEvaluation::at_boundary() is not implemented for face evaluators
            // reinited via the cell-centric interface (reinit(cell_batch_index, face_number)) --
            // it always throws ExcNotImplemented() for this access mode. We therefore have to
            // query boundary-ness directly from the MatrixFree object instead.
            //
            // A single cell batch vectorizes over several (unrelated) cells, so its lanes can mix
            // cells that are adjacent to a domain boundary on this face with cells that are not
            // (e.g. the last batch of a 1d mesh whose length isn't a multiple of the vectorization
            // width). FEFaceEvaluation's cell-centric exterior/neighbor access does not support
            // such mixed batches -- gathering neighbor values would dereference a null pointer for
            // the lanes without a neighbor. We therefore conservatively fall back to the
            // boundary-value branch (own cell value) for the whole batch as soon as any active
            // lane has no neighbor on this face. Note: This is wrong and we need to find a better
            // solution here, but it works for now since we don't have any domain boundaries in our
            // test cases.
            const auto &matrix_free = cell_evaluator.get_matrix_free();
            const std::array<dealii::types::boundary_id, dealii::VectorizedArray<number>::size()>
              boundary_ids = matrix_free.get_faces_by_cells_boundary_id(cell_batch_index, face);
            const unsigned int n_active_lanes =
              matrix_free.n_active_entries_per_cell_batch(cell_batch_index);

            const bool is_at_boundary =
              std::any_of(boundary_ids.begin(),
                          boundary_ids.begin() + n_active_lanes,
                          [](const dealii::types::boundary_id id) {
                            return id != dealii::numbers::internal_face_boundary_id;
                          });

            // TODO: We need a better solution here
            if (is_at_boundary)
              {
                inner_face_evaluator.gather_evaluate(solution, dealii::EvaluationFlags::values);

                for (const unsigned int q : inner_face_evaluator.quadrature_point_indices())
                  {
                    const ValueType w_inner = inner_face_evaluator.get_value(q);
                    const dealii::Point<dim, VectorizedArrayType> &location =
                      inner_face_evaluator.quadrature_point(q);
                    const dealii::Tensor<1, dim, VectorizedArrayType> &normal =
                      inner_face_evaluator.normal_vector(q);

                    ValueType w_boundary;

                    for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                      {
                        dealii::Tensor<1, n_components, number> w_boundary_lane;
                        if (boundary_ids[lane] != dealii::numbers::internal_face_boundary_id)
                          {
                            dealii::Point<dim, number>     location_lane;
                            dealii::Tensor<1, dim, number> normal_lane;
                            for (unsigned int d = 0; d < dim; ++d)
                              {
                                location_lane[d] = location[d][lane];
                                normal_lane[d]   = normal[d][lane];
                              }

                            dealii::Tensor<1, n_components, number> w_inner_lane;
                            for (unsigned int c = 0; c < n_components; ++c)
                              w_inner_lane[c] = w_inner[c][lane];

                            const dealii::types::boundary_id boundary_id = boundary_ids[lane];

                            w_boundary_lane = get_boundary_value(location_lane,
                                                                 normal_lane,
                                                                 boundary_id,
                                                                 w_inner_lane);
                          }
                        else
                          {
                            for (unsigned int c = 0; c < n_components; ++c)
                              w_boundary_lane[c] = w_inner[c][lane];
                          }

                        for (unsigned int c = 0; c < n_components; ++c)
                          w_boundary[c][lane] = w_boundary_lane[c];
                      }

                    subcell_values.push_back(w_boundary);
                  }
              }
            else
              {
                outer_face_evaluator.gather_evaluate(solution, dealii::EvaluationFlags::values);
                for (const unsigned int q : outer_face_evaluator.quadrature_point_indices())
                  {
                    subcell_values.push_back(outer_face_evaluator.get_value(q));
                  }
              }
          }
      }

      ValueType
      neighbor_value(std::array<int, dim> cartesian_index, const unsigned int face_index) const
      {
        const unsigned int direction = face_index / 2;
        const unsigned int side      = face_index % 2;
        cartesian_index[direction] += ((side == 0) ? -1 : 1);
        return subcell_values[get_internal_subcell_index(cartesian_index)];
      }

      ValueType
      get_value(const std::array<int, dim> &cartesian_index) const
      {
        return subcell_values[get_internal_subcell_index(cartesian_index)];
      }

      void
      set_value(const std::array<int, dim> &cartesian_index, const ValueType &value)
      {
        subcell_values[get_internal_subcell_index(cartesian_index)] = value;
      }

    private:
      /// The underlying vector of subcell values, including the neighboring subcell values needed
      /// to compute fluxes across the cell boundary. The vector is of size n_q_points_1d^dim +
      /// 2*dim * n_q_points_1d^(dim-1), i.e. it contains the values of all subcells of the cell,
      /// plus one layer of neighboring subcell values on each side. The subcell values are stored
      /// in lexicographic order, with the first index varying fastest. The neighboring subcell
      /// values are face wise starting with face index 0 and going up to face index 2*dim-1.
      std::vector<ValueType> subcell_values;

      const unsigned int n_q_points_1d;

      std::size_t
      get_internal_subcell_index(const std::array<int, dim> &cartesian_index) const
      {
        // Identify whether we are addressing an interior subcell (all indices in
        // [0, n_q_points_1d)) or a face-adjacent neighbor subcell (exactly one index is -1 or
        // n_q_points_1d). Corner/diagonal neighbors are not stored, see class documentation.
        int direction = -1;
        int side      = -1;
        for (std::size_t d = 0; d < dim; ++d)
          {
            const int index_in_direction = cartesian_index[d];

            AssertThrow(
              index_in_direction >= -1 and index_in_direction <= static_cast<int>(n_q_points_1d),
              dealii::ExcMessage("Cartesian index out of bounds for subcell values access."));

            if (index_in_direction == -1 or index_in_direction == static_cast<int>(n_q_points_1d))
              {
                AssertThrow(direction == -1,
                            dealii::ExcMessage("Diagonal (corner) neighbor subcell values are "
                                               "not supported."));
                direction = static_cast<int>(d);
                side      = (index_in_direction == -1) ? 0 : 1;
              }
          }

        std::size_t n_interior_subcells = dealii::Utilities::fixed_power<dim>(n_q_points_1d);

        if (direction == -1)
          {
            // interior subcell: plain lexicographic index into the first block
            std::size_t lexicographic_index = 0;
            std::size_t stride              = 1;
            for (std::size_t d = 0; d < dim; ++d)
              {
                lexicographic_index += stride * cartesian_index[d];
                stride *= n_q_points_1d;
              }

            return lexicographic_index;
          }

        // face-adjacent neighbor subcell: located in the block for the corresponding face,
        // indexed lexicographically by the remaining (tangential) cartesian indices
        const unsigned int face = 2 * direction + side;

        std::size_t face_local_index = 0;
        std::size_t stride           = 1;
        for (std::size_t d = 0; d < dim; ++d)
          {
            if (static_cast<int>(d) == direction)
              continue;

            face_local_index += stride * cartesian_index[d];
            stride *= n_q_points_1d;
          }

        const std::size_t n_subcells_per_face =
          dealii::Utilities::fixed_power<dim - 1>(n_q_points_1d);

        return n_interior_subcells + face * n_subcells_per_face + face_local_index;
      }
    };

    std::array<int, dim>
    get_cartesian_from_lexicographic_q_index(const std::size_t q) const
    {
      const std::size_t n_q_points_1d =
        matrix_free_context.mf.get_quadrature(matrix_free_context.quad_idx)
          .get_tensor_basis()[0]
          .size();

      std::array<int, dim> cartesian_index;
      std::size_t          remaining_q = q;
      for (std::size_t d = 0; d < dim; ++d)
        {
          cartesian_index[d] = remaining_q % n_q_points_1d;
          remaining_q /= n_q_points_1d;
        }
      return cartesian_index;
    }

    unsigned int
    inter_cell_numerical_admissibility_marking(
      const VectorType                                       &solution,
      dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
      std::function<dealii::Tensor<1, n_components, number>(
        const dealii::Point<dim, number> &,
        const dealii::Tensor<1, dim, number> &,
        dealii::types::boundary_id,
        const dealii::Tensor<1, n_components, number> &)>     get_boundary_value) const;

    unsigned int
    local_cell_numerical_admissibility_marking(
      const VectorType                                       &solution,
      dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
      std::function<dealii::Tensor<1, n_components, number>(
        const dealii::Point<dim, number> &,
        const dealii::Tensor<1, dim, number> &,
        dealii::types::boundary_id,
        const dealii::Tensor<1, n_components, number> &)>     get_boundary_value) const;

    unsigned int
    physical_admissibility_marking(
      const VectorType                                                        &solution,
      dealii::AlignedVector<dealii::VectorizedArray<number>>                  &marked_cells_dst,
      const std::function<dealii::VectorizedArray<number>(const ValueType &)> &admissibility_check)
      const;
  };


  template <int dim, int n_components, typename number>
  void
  Limiter<dim, n_components, number>::reinit()
  {
    troubled_cells.resize(matrix_free_context.mf.n_cell_batches());
    matrix_free_context.mf.initialize_dof_vector(previous_time_solution);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::physical_admissibility_marking(
    const VectorType                                                        &solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>>                  &marked_cells_dst,
    const std::function<dealii::VectorizedArray<number>(const ValueType &)> &admissibility_check)
    const
  {
    Assert(admissibility_check,
           dealii::ExcMessage(
             "Physical admissibility check function must be provided for physical admissibility "
             "marking."));

    unsigned int cells_marked = 0;

    std::function<void(const dealii::MatrixFree<dim, number> &,
                       dealii::AlignedVector<dealii::VectorizedArray<number>> &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      mark_troubled_cells =
        [&](const dealii::MatrixFree<dim, number>                  &matrix_free,
            dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells,
            const VectorType                                       &current_solution,
            const std::pair<unsigned int, unsigned int>            &cell_range) {
          FECellIntegrator<dim, n_components, number> cell_evaluator(matrix_free,
                                                                     matrix_free_context.dof_idx,
                                                                     matrix_free_context.quad_idx);

          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              cell_evaluator.reinit(cell);
              cell_evaluator.gather_evaluate(current_solution, dealii::EvaluationFlags::values);

              dealii::VectorizedArray<number> local_troubled_cells = 0.;
              for (const unsigned int subcell : cell_evaluator.quadrature_point_indices())
                {
                  local_troubled_cells =
                    dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                      admissibility_check(cell_evaluator.get_value(subcell)),
                      dealii::VectorizedArray<number>(0.),
                      dealii::VectorizedArray<number>(1.),
                      local_troubled_cells);
                }

              if (matrix_free.n_active_entries_per_cell_batch(cell) <
                  dealii::VectorizedArray<number>::size())
                {
                  // Non-active entries shall not be marked as troubled, so we mask them out here.
                  for (unsigned int lane = matrix_free.n_active_entries_per_cell_batch(cell);
                       lane < dealii::VectorizedArray<number>::size();
                       ++lane)
                    {
                      local_troubled_cells[lane] = 0;
                    }
                }

              cells_marked += local_troubled_cells.sum();

              local_troubled_cells = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                local_troubled_cells,
                dealii::VectorizedArray<number>(1.),
                dealii::VectorizedArray<number>(1.),
                cell_evaluator.read_cell_data(marked_cells));
              cell_evaluator.set_cell_data(marked_cells, local_troubled_cells);
            }
        };

    matrix_free_context.mf.cell_loop(mark_troubled_cells, marked_cells_dst, solution);
    return dealii::Utilities::MPI::sum(cells_marked, MPI_COMM_WORLD);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::local_cell_numerical_admissibility_marking(
    const VectorType                                       &solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
    std::function<dealii::Tensor<1, n_components, number>(
      const dealii::Point<dim, number> &,
      const dealii::Tensor<1, dim, number> &,
      dealii::types::boundary_id,
      const dealii::Tensor<1, n_components, number> &)>     get_boundary_value) const
  {
    unsigned int cells_marked = 0;

    std::function<void(const dealii::MatrixFree<dim, number> &,
                       dealii::AlignedVector<dealii::VectorizedArray<number>> &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      mark_troubled_cells =
        [&](const dealii::MatrixFree<dim, number>                  &matrix_free,
            dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells,
            const VectorType                                       &current_solution,
            const std::pair<unsigned int, unsigned int>            &cell_range) {
          FECellIntegrator<dim, n_components, number> cell_evaluator_new(
            matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
          FECellIntegrator<dim, n_components, number> cell_evaluator_old(
            matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
          FEFaceIntegrator<dim, n_components, number> outer_face_evaluator(
            matrix_free, false, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
          FEFaceIntegrator<dim, n_components, number> inner_face_evaluator(
            matrix_free, true, matrix_free_context.dof_idx, matrix_free_context.quad_idx);

          const VectorizedArrayType tol(1e-5); // TODO: How to deal with this
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              cell_evaluator_new.reinit(cell);
              cell_evaluator_old.reinit(cell);
              cell_evaluator_new.gather_evaluate(current_solution, dealii::EvaluationFlags::values);
              cell_evaluator_old.gather_evaluate(previous_time_solution,
                                                 dealii::EvaluationFlags::values);

              CartesianIndexedSubcellValues subcell_average_values(
                get_boundary_value,
                cell_evaluator_old,
                outer_face_evaluator,
                inner_face_evaluator,
                previous_time_solution,
                cell,
                matrix_free.get_quadrature(matrix_free_context.quad_idx)
                  .get_tensor_basis()[0]
                  .size());


              dealii::VectorizedArray<number> local_troubled_cells = 0.;
              for (const unsigned int subcell : cell_evaluator_new.quadrature_point_indices())
                {
                  std::array<int, dim> cartesian_index =
                    get_cartesian_from_lexicographic_q_index(subcell);

                  const ValueType w_old = cell_evaluator_old.get_value(subcell);

                  ValueType min_neighbor_values = w_old;
                  ValueType max_neighbor_values = w_old;
                  for (unsigned int subcell_face = 0;
                       subcell_face < dealii::GeometryInfo<dim>::faces_per_cell;
                       ++subcell_face)
                    {
                      min_neighbor_values =
                        elementwise_min(min_neighbor_values,
                                        subcell_average_values.neighbor_value(cartesian_index,
                                                                              subcell_face));
                      max_neighbor_values =
                        elementwise_max(max_neighbor_values,
                                        subcell_average_values.neighbor_value(cartesian_index,
                                                                              subcell_face));
                    }

                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      local_troubled_cells =
                        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                          cell_evaluator_new.get_value(subcell)[c],
                          min_neighbor_values[c] - tol,
                          1.,
                          local_troubled_cells);

                      local_troubled_cells =
                        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                          cell_evaluator_new.get_value(subcell)[c],
                          max_neighbor_values[c] + tol,
                          1.,
                          local_troubled_cells);
                    }
                }

              if (matrix_free.n_active_entries_per_cell_batch(cell) <
                  dealii::VectorizedArray<number>::size())
                {
                  // Non-active entries shall not be marked as troubled, so we mask them out here.
                  for (unsigned int lane = matrix_free.n_active_entries_per_cell_batch(cell);
                       lane < dealii::VectorizedArray<number>::size();
                       ++lane)
                    {
                      local_troubled_cells[lane] = 0;
                    }
                }

              cells_marked += local_troubled_cells.sum();

              local_troubled_cells = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                local_troubled_cells,
                dealii::VectorizedArray<number>(1),
                dealii::VectorizedArray<number>(1),
                cell_evaluator_new.read_cell_data(marked_cells));
              cell_evaluator_new.set_cell_data(marked_cells, local_troubled_cells);
            }
        };

    matrix_free_context.mf.cell_loop(mark_troubled_cells, marked_cells_dst, solution);
    return dealii::Utilities::MPI::sum(cells_marked, MPI_COMM_WORLD);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::inter_cell_numerical_admissibility_marking(
    const VectorType                                       &solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
    std::function<dealii::Tensor<1, n_components, number>(
      const dealii::Point<dim, number> &,
      const dealii::Tensor<1, dim, number> &,
      dealii::types::boundary_id,
      const dealii::Tensor<1, n_components, number> &)>     get_boundary_value) const
  {
    unsigned int cells_marked = 0;

    std::function<void(const dealii::MatrixFree<dim, number> &,
                       DistributedCellData<dim,
                                           std::pair<dealii::Tensor<1, n_components, number>,
                                                     dealii::Tensor<1, n_components, number>>> &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      compute_min_max_subcell_values =
        [dof_idx = matrix_free_context.dof_idx, quad_idx = matrix_free_context.quad_idx](
          const dealii::MatrixFree<dim, number> &matrix_free,
          DistributedCellData<dim,
                              std::pair<dealii::Tensor<1, n_components, number>,
                                        dealii::Tensor<1, n_components, number>>>
                                                      &min_max_subcell_values,
          const VectorType                            &old_solution,
          const std::pair<unsigned int, unsigned int> &cell_range) {
          FECellIntegrator<dim, n_components, number> cell_evaluator_old(matrix_free,
                                                                         dof_idx,
                                                                         quad_idx);

          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              cell_evaluator_old.reinit(cell);
              cell_evaluator_old.gather_evaluate(old_solution, dealii::EvaluationFlags::values);

              ValueType min_subcell_values = cell_evaluator_old.get_value(0);
              ValueType max_subcell_values = cell_evaluator_old.get_value(0);
              for (const unsigned int q : cell_evaluator_old.quadrature_point_indices())
                {
                  const ValueType subcell_values = cell_evaluator_old.get_value(q);
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      min_subcell_values[c] =
                        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                          subcell_values[c],
                          min_subcell_values[c],
                          subcell_values[c],
                          min_subcell_values[c]);
                      max_subcell_values[c] =
                        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                          subcell_values[c],
                          max_subcell_values[c],
                          subcell_values[c],
                          max_subcell_values[c]);
                    }
                }

              const auto cells = cells_in_cell_batch(matrix_free, cell);
              for (unsigned int lane = 0; lane < matrix_free.n_active_entries_per_cell_batch(cell);
                   ++lane)
                {
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      min_max_subcell_values[cells[lane]->active_cell_index()].first[c] =
                        min_subcell_values[c][lane];
                      min_max_subcell_values[cells[lane]->active_cell_index()].second[c] =
                        max_subcell_values[c][lane];
                    }
                }
            }
        };

    // Step 1: compute cell-wise min and max subcell values at previous time step
    DistributedCellData<
      dim,
      std::pair<dealii::Tensor<1, n_components, number>, dealii::Tensor<1, n_components, number>>>
      min_max_subcell_values(
        matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx).get_triangulation());
    matrix_free_context.mf.cell_loop(compute_min_max_subcell_values,
                                     min_max_subcell_values,
                                     previous_time_solution);
    min_max_subcell_values.update_ghost_values();

    // Step 2: Compute NAD criteria
    std::function<void(const dealii::MatrixFree<dim, number> &,
                       dealii::AlignedVector<dealii::VectorizedArray<number>> &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      mark_cells = [&](const dealii::MatrixFree<dim, number>                  &matrix_free,
                       dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells,
                       const VectorType                                       &current_solution,
                       const std::pair<unsigned int, unsigned int>            &cell_range) {
        FECellIntegrator<dim, n_components, number> cell_evaluator_new(
          matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
        FEFaceIntegrator<dim, n_components, number> face_evaluator(matrix_free,
                                                                   true,
                                                                   matrix_free_context.dof_idx,
                                                                   matrix_free_context.quad_idx);

        const VectorizedArrayType tol(1e-5); // TODO: Hwo to deal with this
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            cell_evaluator_new.reinit(cell);
            cell_evaluator_new.gather_evaluate(current_solution, dealii::EvaluationFlags::values);

            const auto         cells          = cells_in_cell_batch(matrix_free, cell);
            const unsigned int n_active_lanes = matrix_free.n_active_entries_per_cell_batch(cell);
            ValueType          min_subcell_values;
            ValueType          max_subcell_values;
            for (unsigned int c = 0; c < n_components; ++c)
              {
                min_subcell_values[c] = std::numeric_limits<number>::max();
                max_subcell_values[c] = std::numeric_limits<number>::lowest();
              }

            // Precompute a JxW-weighted average of the boundary condition over each face of the
            // cell batch that lies on the domain boundary (for at least one lane). This average
            // is used below as the "virtual neighbor" value on boundary faces, mirroring how a
            // real neighbor contributes a single min/max pair rather than pointwise values.
            std::array<ValueType, dealii::GeometryInfo<dim>::faces_per_cell>
              averaged_boundary_values{};

            for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                const std::array<dealii::types::boundary_id, VectorizedArrayType::size()>
                  boundary_ids = matrix_free.get_faces_by_cells_boundary_id(cell, face);

                const bool any_lane_at_boundary =
                  std::any_of(boundary_ids.begin(),
                              boundary_ids.begin() + n_active_lanes,
                              [](const dealii::types::boundary_id id) {
                                return id != dealii::numbers::internal_face_boundary_id;
                              });
                if (!any_lane_at_boundary)
                  continue;

                face_evaluator.reinit(cell, face);
                face_evaluator.gather_evaluate(current_solution, dealii::EvaluationFlags::values);

                ValueType           weighted_sum{};
                VectorizedArrayType weight_sum = 0.;

                for (const unsigned int q : face_evaluator.quadrature_point_indices())
                  {
                    const ValueType w_inner = face_evaluator.get_value(q);
                    const dealii::Point<dim, VectorizedArrayType> &location =
                      face_evaluator.quadrature_point(q);
                    const dealii::Tensor<1, dim, VectorizedArrayType> &normal =
                      face_evaluator.normal_vector(q);
                    const VectorizedArrayType JxW = face_evaluator.JxW(q);

                    for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                      {
                        if (boundary_ids[lane] == dealii::numbers::internal_face_boundary_id)
                          continue;

                        dealii::Point<dim, number>     location_lane;
                        dealii::Tensor<1, dim, number> normal_lane;
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            location_lane[d] = location[d][lane];
                            normal_lane[d]   = normal[d][lane];
                          }

                        dealii::Tensor<1, n_components, number> w_inner_lane;
                        for (unsigned int c = 0; c < n_components; ++c)
                          w_inner_lane[c] = w_inner[c][lane];

                        const dealii::Tensor<1, n_components, number> w_boundary_lane =
                          get_boundary_value(location_lane,
                                             normal_lane,
                                             boundary_ids[lane],
                                             w_inner_lane);

                        for (unsigned int c = 0; c < n_components; ++c)
                          weighted_sum[c][lane] += w_boundary_lane[c] * JxW[lane];
                        weight_sum[lane] += JxW[lane];
                      }
                  }

                for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                  {
                    if (boundary_ids[lane] == dealii::numbers::internal_face_boundary_id)
                      continue;

                    for (unsigned int c = 0; c < n_components; ++c)
                      averaged_boundary_values[face][c][lane] =
                        weighted_sum[c][lane] / weight_sum[lane];
                  }
              }

            for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
              {
                // own cells
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    min_subcell_values[c][lane] =
                      std::min(min_subcell_values[c][lane],
                               min_max_subcell_values[cells[lane]->active_cell_index()].first[c]);
                    max_subcell_values[c][lane] =
                      std::max(max_subcell_values[c][lane],
                               min_max_subcell_values[cells[lane]->active_cell_index()].second[c]);
                  }

                // neighbour cells
                for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell;
                     ++face)
                  {
                    if (!cells[lane]->at_boundary(face))
                      {
                        const auto neighbor_cell = cells[lane]->neighbor(face);
                        for (unsigned int c = 0; c < n_components; ++c)
                          {
                            min_subcell_values[c][lane] = std::min(
                              min_subcell_values[c][lane],
                              min_max_subcell_values[neighbor_cell->active_cell_index()].first[c]);
                            max_subcell_values[c][lane] = std::max(
                              max_subcell_values[c][lane],
                              min_max_subcell_values[neighbor_cell->active_cell_index()].second[c]);
                          }
                      }
                    else
                      {
                        // Domain boundary: no neighbor cell exists, so use the JxW-weighted
                        // average of the prescribed boundary value over this face (computed
                        // above) as a stand-in "virtual neighbor" value.
                        for (unsigned int c = 0; c < n_components; ++c)
                          {
                            min_subcell_values[c][lane] =
                              std::min(min_subcell_values[c][lane],
                                       averaged_boundary_values[face][c][lane]);
                            max_subcell_values[c][lane] =
                              std::max(max_subcell_values[c][lane],
                                       averaged_boundary_values[face][c][lane]);
                          }
                      }
                  }
              }

            dealii::VectorizedArray<number> local_troubled_cells = 0;
            for (const unsigned int q : cell_evaluator_new.quadrature_point_indices())
              {
                const ValueType subcell_values = cell_evaluator_new.get_value(q);
                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    local_troubled_cells =
                      dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                        subcell_values[c],
                        min_subcell_values[c] - tol,
                        dealii::VectorizedArray<number>(1),
                        local_troubled_cells);
                    local_troubled_cells =
                      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                        subcell_values[c],
                        max_subcell_values[c] + tol,
                        dealii::VectorizedArray<number>(1),
                        local_troubled_cells);
                  }
              }

            if (matrix_free.n_active_entries_per_cell_batch(cell) <
                dealii::VectorizedArray<number>::size())
              {
                // Non-active entries shall not be marked as troubled, so we mask them out here.
                for (unsigned int lane = matrix_free.n_active_entries_per_cell_batch(cell);
                     lane < dealii::VectorizedArray<number>::size();
                     ++lane)
                  {
                    local_troubled_cells[lane] = 0;
                  }
              }

            cells_marked += local_troubled_cells.sum();

            local_troubled_cells = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              local_troubled_cells,
              dealii::VectorizedArray<number>(1),
              dealii::VectorizedArray<number>(1),
              cell_evaluator_new.read_cell_data(marked_cells));
            cell_evaluator_new.set_cell_data(marked_cells, local_troubled_cells);
          }
      };

    matrix_free_context.mf.cell_loop(mark_cells, marked_cells_dst, solution);
    return dealii::Utilities::MPI::sum(cells_marked, MPI_COMM_WORLD);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::mark_cells_for_limiting(
    const VectorType                                                        &solution,
    std::function<dealii::Tensor<1, n_components, number>(
      const dealii::Point<dim, number> &,
      const dealii::Tensor<1, dim, number> &,
      dealii::types::boundary_id,
      const dealii::Tensor<1, n_components, number> &)>                      get_boundary_value,
    const std::function<dealii::VectorizedArray<number>(const ValueType &)> &admissibility_check)
  {
    // TODO 1: Consider local smooth extrema
    // TODO 2: Check for physical admissibility (e.g. positivity of density and pressure)

    unsigned int cells_marked = 0;
    for (dealii::VectorizedArray<number> &troubled_cells_batch : troubled_cells)
      troubled_cells_batch = 0.;

    if (Utils::contains(limiter_data.troubled_cell_marking_types,
                        TroubledCellMarkingType::inter_cell_numerical_admissibility))
      cells_marked +=
        inter_cell_numerical_admissibility_marking(solution, troubled_cells, get_boundary_value);

    if (Utils::contains(limiter_data.troubled_cell_marking_types,
                        TroubledCellMarkingType::local_cell_numerical_admissibility))
      cells_marked +=
        local_cell_numerical_admissibility_marking(solution, troubled_cells, get_boundary_value);

    if (Utils::contains(limiter_data.troubled_cell_marking_types,
                        TroubledCellMarkingType::physical_admissibility))
      cells_marked += physical_admissibility_marking(solution, troubled_cells, admissibility_check);

    return cells_marked;
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::apply_limiting(
    const number                                                              time_step,
    const std::function<FluxType(const ValueType &w_m, const ValueType &w_p)> numerical_flux,
    std::function<dealii::Tensor<1, n_components, number>(
      const dealii::Point<dim, number> &,
      const dealii::Tensor<1, dim, number> &,
      dealii::types::boundary_id,
      const dealii::Tensor<1, n_components, number> &)>                       get_boundary_value,
    VectorType                                                               &limited_solution,
    const VectorType                                                         &solution,
    const std::function<dealii::VectorizedArray<number>(const ValueType &)>  &admissibility_check)
  {
    if (limited_solution.has_ghost_elements())
      limited_solution.zero_out_ghost_values();

    const unsigned int n_cells_to_limit =
      mark_cells_for_limiting(solution, get_boundary_value, admissibility_check);

    std::cout << "Number of cells to limit: " << n_cells_to_limit << std::endl;

    std::function<void(const dealii::MatrixFree<dim, number> &,
                       VectorType &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      mf_limiter_loop = [&](const dealii::MatrixFree<dim, number>       &matrix_free,
                            VectorType                                  &dst,
                            const VectorType                            &src,
                            const std::pair<unsigned int, unsigned int> &cell_range) {
        FECellIntegrator<dim, n_components, number> cell_evaluator(matrix_free,
                                                                   matrix_free_context.dof_idx,
                                                                   matrix_free_context.quad_idx);
        FECellIntegrator<dim, n_components, number> cell_evaluator_old(
          matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
        FEFaceIntegrator<dim, n_components, number> outer_face_evaluator(
          matrix_free, false, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
        FEFaceIntegrator<dim, n_components, number> inner_face_evaluator(
          matrix_free, true, matrix_free_context.dof_idx, matrix_free_context.quad_idx);

        dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, number>
          inverse(cell_evaluator);

        std::vector<VectorizedArrayType> fv_solutions(
          n_components * cell_evaluator.quadrature_point_indices().size());

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            cell_evaluator.reinit(cell);
            const dealii::VectorizedArray<number> local_troubled_cells =
              cell_evaluator.read_cell_data(troubled_cells);

            // We only perform the computations if at least one of the lanes in the cell is marked
            // as troubled
            if (local_troubled_cells.sum() == 0)
              continue;

            cell_evaluator_old.reinit(cell);
            cell_evaluator.gather_evaluate(src, dealii::EvaluationFlags::values);
            cell_evaluator_old.gather_evaluate(previous_time_solution,
                                               dealii::EvaluationFlags::values);

            // For defining the subcells we follow the approach by Sonntag et al. and define the
            // subcells by taking each quadrature point as a centre of a subcell resulting in
            // n_q_points subcells per cell. Hence, looping over all subcells is equivalent to
            // looping over all quadrature points.
            //
            // The finite volume update must be recomputed entirely from the previous time (stage)
            // level solution -- both the cell's own subcell values and its neighbors' -- so that
            // it yields a robust, bound-preserving replacement value. Basing it on the (possibly
            // already oscillatory) new solution `src` would make the correction a small
            // perturbation on top of the polluted value instead of an independent low-order
            // update.
            CartesianIndexedSubcellValues subcell_average_values(
              get_boundary_value,
              cell_evaluator_old,
              outer_face_evaluator,
              inner_face_evaluator,
              previous_time_solution,
              cell,
              matrix_free.get_quadrature(matrix_free_context.quad_idx)
                .get_tensor_basis()[0]
                .size());

            for (const unsigned int subcell : cell_evaluator.quadrature_point_indices())
              {
                std::array<int, dim> cartesian_index =
                  get_cartesian_from_lexicographic_q_index(subcell);

                const ValueType           w_old = cell_evaluator_old.get_value(subcell);
                ValueType                 fv_subcell_average = w_old;
                const VectorizedArrayType prefactor =
                  VectorizedArrayType(time_step) / cell_evaluator.JxW(subcell);
                for (unsigned int subcell_face = 0;
                     subcell_face < dealii::GeometryInfo<dim>::faces_per_cell;
                     ++subcell_face)
                  {
                    dealii::Tensor<1, dim, VectorizedArrayType> normal;
                    normal[subcell_face / 2] = (subcell_face % 2 == 0) ? -1. : 1.;

                    // The subcell face separating this subcell from its neighbor  has, in
                    // reference coordinates, an area given by the product of the tangential (i.e.
                    // all but the face-normal direction) subcell widths, which in turn equal the
                    // 1d quadrature weights at the corresponding cartesian index.
                    const unsigned int          direction = subcell_face / 2;
                    const dealii::Quadrature<1> quadrature_1d =
                      matrix_free.get_quadrature(matrix_free_context.quad_idx)
                        .get_tensor_basis()[0];

                    VectorizedArrayType face_size = VectorizedArrayType(1.);
                    for (unsigned int d = 0; d < dim; ++d)
                      if (d != direction)
                        face_size *= VectorizedArrayType(
                          quadrature_1d.weight(static_cast<unsigned int>(cartesian_index[d])));

                    // TODO: We need the Jacobian

                    // numerical_flux(w_left, w_right) approximates the physical flux at the
                    // interface using the state ordered by the *global* coordinate direction, not
                    // by "own vs. neighbor". On the negative-direction (e.g. west) face, the
                    // neighbor is the left state and our own subcell is the right state; on the
                    // positive-direction (e.g. east) face it's the other way around. Because the
                    // Rusanov/LLF dissipation term is not symmetric in its two arguments, getting
                    // this ordering wrong on the negative-direction faces flips the sign of the
                    // dissipation there and makes the update unstable.
                    const bool      is_negative_side = (subcell_face % 2 == 0);
                    const ValueType neighbor =
                      subcell_average_values.neighbor_value(cartesian_index, subcell_face);
                    const FluxType flux = is_negative_side ? numerical_flux(neighbor, w_old) :
                                                             numerical_flux(w_old, neighbor);

                    for (unsigned int c = 0; c < n_components; ++c)
                      for (unsigned int d = 0; d < dim; ++d)
                        fv_subcell_average[c] -= prefactor * face_size * flux[c][d] * normal[d];
                  }

                const ValueType w_new = cell_evaluator.get_value(subcell);
                for (unsigned int c = 0; c < n_components; ++c)
                  fv_solutions[c * cell_evaluator.quadrature_point_indices().size() + subcell] =
                    dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                      local_troubled_cells,
                      dealii::VectorizedArray<number>(1),
                      fv_subcell_average[c],
                      w_new[c]);
              }
            inverse.transform_from_q_points_to_basis(n_components,
                                                     fv_solutions.data(),
                                                     cell_evaluator.begin_dof_values());
            cell_evaluator.set_dof_values(dst);
          }
      };

    matrix_free_context.mf.loop_cell_centric(mf_limiter_loop, limited_solution, solution);
    return n_cells_to_limit;
  }

  /**
   * Ready functions
   */

  template <int dim, int n_components, typename number>
  void
  Limiter<dim, n_components, number>::prepare_for_limiting(const VectorType &solution)
  {
    previous_time_solution = solution;
  }

  template <int dim, int n_components, typename number>
  void
  Limiter<dim, n_components, number>::attach_to_data_out(
    GenericDataOut<dim, number> &data_out) const
  {
    // Since GenericDataOut expects a Vector, we need to convert the AlignedVector of
    // VectorizedArray<int> to a Vector<int> which is stored internally in the this class due to
    // the fact that GenericDataOut stores a pointer to the vector internally.
    troubled_cells_output.reinit(matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx)
                                   .get_triangulation()
                                   .n_active_cells());

    for (unsigned int cell_batch = 0; cell_batch < matrix_free_context.mf.n_cell_batches();
         ++cell_batch)
      for (unsigned int lane = 0;
           lane < matrix_free_context.mf.n_active_entries_per_cell_batch(cell_batch);
           ++lane)
        {
          const auto cell_iterator =
            matrix_free_context.mf.get_cell_iterator(cell_batch, lane, matrix_free_context.dof_idx);

          troubled_cells_output[cell_iterator->active_cell_index()] =
            troubled_cells[cell_batch][lane];
        }

    data_out.add_element_wise_data_vector(troubled_cells_output, "limited_cells");
  }
} // namespace MeltPoolDG::Utilities
