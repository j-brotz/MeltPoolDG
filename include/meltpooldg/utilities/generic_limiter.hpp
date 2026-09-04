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

#include "meltpooldg/utilities/fe_subcell_evaluation.hpp"
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/fe_subcell_evaluation.templates.hpp>
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
      std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                              dealii::types::boundary_id,
                              const ValueType &)>                              get_boundary_value,
      const std::function<dealii::VectorizedArray<number>(const ValueType &)> &admissibility_check);

    void
    prepare_for_limiting(const VectorType &solution);

    unsigned int
    apply_limiting(
      const number                                                              time_step,
      const std::function<FluxType(const ValueType &w_m, const ValueType &w_p)> numerical_flux,
      std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                              dealii::types::boundary_id,
                              const ValueType &)>                               get_boundary_value,
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

    unsigned int
    inter_cell_numerical_admissibility_marking(
      const VectorType                                       &solution,
      dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
      std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                              dealii::types::boundary_id,
                              const ValueType &)>             get_boundary_value) const;

    unsigned int
    local_cell_numerical_admissibility_marking(
      const VectorType                                       &solution,
      dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
      std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                              dealii::types::boundary_id,
                              const ValueType &)>             get_boundary_value) const;

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

    matrix_free_context.mf.loop_cell_centric(mark_troubled_cells, marked_cells_dst, solution);
    return dealii::Utilities::MPI::sum(cells_marked, MPI_COMM_WORLD);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::local_cell_numerical_admissibility_marking(
    const VectorType                                       &solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
    std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                            dealii::types::boundary_id,
                            const ValueType &)>             get_boundary_value) const
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
          FESubcellEvaluation<dim, n_components, number> new_subcells(matrix_free,
                                                                      matrix_free_context.dof_idx,
                                                                      matrix_free_context.quad_idx);
          FESubcellEvaluation<dim, n_components, number> old_subcells(matrix_free,
                                                                      matrix_free_context.dof_idx,
                                                                      matrix_free_context.quad_idx);

          const VectorizedArrayType tol(1e-5); // TODO: How to deal with this
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              new_subcells.reinit(cell);
              old_subcells.reinit(cell);
              new_subcells.gather_evaluate(current_solution, get_boundary_value);
              old_subcells.gather_evaluate(previous_time_solution, get_boundary_value);


              dealii::VectorizedArray<number> local_troubled_cells = 0.;
              for (const unsigned int subcell : new_subcells.subcell_indices())
                {
                  const ValueType w_subcell_old = old_subcells.get_subcell_value(subcell);

                  ValueType min_neighbor_values = w_subcell_old;
                  ValueType max_neighbor_values = w_subcell_old;
                  for (unsigned int subcell_face = 0;
                       subcell_face < dealii::GeometryInfo<dim>::faces_per_cell;
                       ++subcell_face)
                    {
                      min_neighbor_values =
                        elementwise_min(min_neighbor_values,
                                        old_subcells.get_subcell_neighbor_value(subcell,
                                                                                subcell_face));
                      max_neighbor_values =
                        elementwise_max(max_neighbor_values,
                                        old_subcells.get_subcell_neighbor_value(subcell,
                                                                                subcell_face));
                    }

                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      local_troubled_cells =
                        dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                          new_subcells.get_subcell_value(subcell)[c],
                          min_neighbor_values[c] - tol,
                          1.,
                          local_troubled_cells);

                      local_troubled_cells =
                        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                          new_subcells.get_subcell_value(subcell)[c],
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
                dealii::VectorizedArray<number>(1.),
                dealii::VectorizedArray<number>(1.),
                new_subcells.read_cell_data(marked_cells));
              new_subcells.set_cell_data(marked_cells, local_troubled_cells);
            }
        };

    matrix_free_context.mf.loop_cell_centric(mark_troubled_cells, marked_cells_dst, solution);
    return dealii::Utilities::MPI::sum(cells_marked, MPI_COMM_WORLD);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::inter_cell_numerical_admissibility_marking(
    const VectorType                                       &solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>> &marked_cells_dst,
    std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                            dealii::types::boundary_id,
                            const ValueType &)>             get_boundary_value) const
  {
    unsigned int cells_marked = 0;

    std::function<void(const dealii::MatrixFree<dim, number> &,
                       DistributedCellData<dim,
                                           std::pair<dealii::Tensor<1, n_components, number>,
                                                     dealii::Tensor<1, n_components, number>>> &,
                       const VectorType &,
                       const std::pair<unsigned int, unsigned int> &)>
      compute_min_max_subcell_values =
        [dof_idx  = matrix_free_context.dof_idx,
         quad_idx = matrix_free_context.quad_idx,
         &get_boundary_value](
          const dealii::MatrixFree<dim, number> &matrix_free,
          DistributedCellData<dim,
                              std::pair<dealii::Tensor<1, n_components, number>,
                                        dealii::Tensor<1, n_components, number>>>
                                                      &min_max_subcell_values,
          const VectorType                            &old_solution,
          const std::pair<unsigned int, unsigned int> &cell_range) {
          FESubcellEvaluation<dim, n_components, number> subcell_evaluator_old(matrix_free,
                                                                               dof_idx,
                                                                               quad_idx);

          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              subcell_evaluator_old.reinit(cell);
              subcell_evaluator_old.gather_evaluate(old_solution, get_boundary_value);

              ValueType min_subcell_values = subcell_evaluator_old.get_value(0);
              ValueType max_subcell_values = subcell_evaluator_old.get_value(0);
              for (const unsigned int subcell : subcell_evaluator_old.subcell_indices())
                {
                  const ValueType subcell_values = subcell_evaluator_old.get_subcell_value(subcell);
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
    matrix_free_context.mf.loop_cell_centric(compute_min_max_subcell_values,
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
        FESubcellEvaluation<dim, n_components, number> subcell_evaluator_new(
          matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
        FEFaceIntegrator<dim, n_components, number> face_evaluator(matrix_free,
                                                                   true,
                                                                   matrix_free_context.dof_idx,
                                                                   matrix_free_context.quad_idx);

        const VectorizedArrayType tol(1e-5); // TODO: Hwo to deal with this
        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            subcell_evaluator_new.reinit(cell);
            subcell_evaluator_new.gather_evaluate(current_solution, get_boundary_value);

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
                    const VectorizedArrayType JxW = face_evaluator.JxW(q);

                    for (unsigned int lane = 0; lane < n_active_lanes; ++lane)
                      {
                        if (boundary_ids[lane] == dealii::numbers::internal_face_boundary_id)
                          continue;

                        dealii::Point<dim, number> location_lane;
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            location_lane[d] = location[d][lane];
                          }

                        dealii::Tensor<1, n_components, number> w_inner_lane;
                        for (unsigned int c = 0; c < n_components; ++c)
                          w_inner_lane[c] = w_inner[c][lane];

                        const dealii::Tensor<1, n_components, number> w_boundary_lane =
                          get_boundary_value(location_lane, boundary_ids[lane], w_inner_lane);

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

            // Check if any subcell values are outside the min/max range of the cell and its
            // neighbors at its old time step. If so, mark the cell as troubled.
            dealii::VectorizedArray<number> local_troubled_cells = 0;
            for (const unsigned int subcell : subcell_evaluator_new.subcell_indices())
              {
                const ValueType subcell_values = subcell_evaluator_new.get_subcell_value(subcell);
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

            // Mark troubled cells in the output vector. We use a compare_and_apply_mask operation
            // to ensure that we only mark cells that are currently unmarked (i.e., have a value of
            // 0) and not accidentally unmark cells that have already been marked as troubled in the
            // given dst vector.
            local_troubled_cells = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              local_troubled_cells,
              dealii::VectorizedArray<number>(1),
              dealii::VectorizedArray<number>(1),
              subcell_evaluator_new.read_cell_data(marked_cells));
            subcell_evaluator_new.set_cell_data(marked_cells, local_troubled_cells);
          }
      };

    matrix_free_context.mf.loop_cell_centric(mark_cells, marked_cells_dst, solution);
    return dealii::Utilities::MPI::sum(cells_marked, MPI_COMM_WORLD);
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::mark_cells_for_limiting(
    const VectorType                                                        &solution,
    std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                            dealii::types::boundary_id,
                            const ValueType &)>                              get_boundary_value,
    const std::function<dealii::VectorizedArray<number>(const ValueType &)> &admissibility_check)
  {
    // TODO 1: Consider local smooth extrema

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
    std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                            dealii::types::boundary_id,
                            const ValueType &)>                               get_boundary_value,
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
        FESubcellEvaluation<dim, n_components, number> subcell_evaluator(
          matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);
        FESubcellEvaluation<dim, n_components, number> subcell_evaluator_old(
          matrix_free, matrix_free_context.dof_idx, matrix_free_context.quad_idx);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            subcell_evaluator.reinit(cell);
            const dealii::VectorizedArray<number> local_troubled_cells =
              subcell_evaluator.read_cell_data(troubled_cells);

            // We only perform the computations if at least one of the lanes in the cell is marked
            // as troubled
            if (local_troubled_cells.sum() == 0)
              continue;

            subcell_evaluator_old.reinit(cell);
            subcell_evaluator.gather_evaluate(src, get_boundary_value);
            subcell_evaluator_old.gather_evaluate(previous_time_solution, get_boundary_value);

            for (const unsigned int subcell : subcell_evaluator.subcell_indices())
              {
                const ValueType           w_old = subcell_evaluator_old.get_subcell_value(subcell);
                ValueType                 fv_subcell_average = w_old;
                const VectorizedArrayType prefactor =
                  VectorizedArrayType(time_step) / subcell_evaluator.subcell_size(subcell);

                for (unsigned int subcell_face = 0;
                     subcell_face < dealii::GeometryInfo<dim>::faces_per_cell;
                     ++subcell_face)
                  {
                    // numerical_flux(w_left, w_right) approximates the physical flux at the
                    // interface using the state ordered by the *global* coordinate direction, not
                    // by "own vs. neighbor". On the negative-direction (e.g. west) face, the
                    // neighbor is the left state and our own subcell is the right state; on the
                    // positive-direction (e.g. east) face it's the other way around.
                    const bool      is_negative_side = (subcell_face % 2 == 0);
                    const ValueType neighbor =
                      subcell_evaluator.get_subcell_neighbor_value(subcell, subcell_face);
                    const FluxType flux = is_negative_side ? numerical_flux(neighbor, w_old) :
                                                             numerical_flux(w_old, neighbor);

                    const dealii::Tensor<1, dim, VectorizedArrayType> normal =
                      subcell_evaluator.subcell_face_normal(subcell, subcell_face);
                    for (unsigned int c = 0; c < n_components; ++c)
                      for (unsigned int d = 0; d < dim; ++d)
                        fv_subcell_average[c] -=
                          prefactor * subcell_evaluator.subcell_face_size(subcell, subcell_face) *
                          flux[c][d] * normal[d];
                  }

                ValueType w_new = subcell_evaluator.get_subcell_value(subcell);
                for (unsigned int c = 0; c < n_components; ++c)
                  w_new[c] = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
                    local_troubled_cells,
                    dealii::VectorizedArray<number>(1),
                    fv_subcell_average[c],
                    w_new[c]);

                subcell_evaluator.submit_subcell_value(subcell, w_new);
              }
            subcell_evaluator.set_dof_values(dst);
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
