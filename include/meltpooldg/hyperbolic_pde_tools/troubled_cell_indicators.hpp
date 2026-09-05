#pragma once

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <meltpooldg/hyperbolic_pde_tools/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/utilities/distributed_cell_data.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <functional>
#include <utility>

namespace MeltPoolDG::HyperbolicPDETools
{
  template <int dim, int n_components, typename number>
  unsigned int
  physical_admissibility_troubled_cell_marking(
    MatrixFreeContext<dim, number>                           &matrix_free_context,
    const dealii::LinearAlgebra::distributed::Vector<number> &solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>>   &marked_cells_dst,
    const std::function<dealii::VectorizedArray<number>(
      const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &)>
      &admissibility_check)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

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
  local_cell_numerical_admissibility_troubled_cell_marking(
    MatrixFreeContext<dim, number>                           &matrix_free_context,
    const dealii::LinearAlgebra::distributed::Vector<number> &previous_solution,
    const dealii::LinearAlgebra::distributed::Vector<number> &current_solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>>   &marked_cells_dst,
    std::function<dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>(
      const dealii::Point<dim, dealii::VectorizedArray<number>> &,
      dealii::types::boundary_id,
      const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &)> get_boundary_value)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using ValueType  = dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>;

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

          const dealii::VectorizedArray<number> tol(1e-5); // TODO: How to deal with this
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              new_subcells.reinit(cell);
              old_subcells.reinit(cell);
              new_subcells.gather_evaluate(current_solution, get_boundary_value);
              old_subcells.gather_evaluate(previous_solution, get_boundary_value);


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

    matrix_free_context.mf.loop_cell_centric(mark_troubled_cells,
                                             marked_cells_dst,
                                             current_solution);
    return dealii::Utilities::MPI::sum(cells_marked, MPI_COMM_WORLD);
  }


  template <int dim, int n_components, typename number>
  unsigned int
  inter_cell_numerical_admissibility_troubled_cell_marking(
    const MatrixFreeContext<dim, number>                     &matrix_free_context,
    const dealii::LinearAlgebra::distributed::Vector<number> &previous_solution,
    const dealii::LinearAlgebra::distributed::Vector<number> &solution,
    dealii::AlignedVector<dealii::VectorizedArray<number>>   &marked_cells_dst,
    std::function<dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>(
      const dealii::Point<dim, dealii::VectorizedArray<number>> &,
      dealii::types::boundary_id,
      const dealii::Tensor<1, n_components, dealii::VectorizedArray<number>> &)> get_boundary_value)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;
    using ValueType  = dealii::Tensor<1, n_components, dealii::VectorizedArray<number>>;

    unsigned int cells_marked = 0;

    std::function<
      void(const dealii::MatrixFree<dim, number> &,
           Utilities::DistributedCellData<dim,
                                          std::pair<dealii::Tensor<1, n_components, number>,
                                                    dealii::Tensor<1, n_components, number>>> &,
           const VectorType &,
           const std::pair<unsigned int, unsigned int> &)>
      compute_min_max_subcell_values =
        [dof_idx  = matrix_free_context.dof_idx,
         quad_idx = matrix_free_context.quad_idx,
         &get_boundary_value](
          const dealii::MatrixFree<dim, number> &matrix_free,
          Utilities::DistributedCellData<dim,
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

              ValueType min_subcell_values = subcell_evaluator_old.get_subcell_value(0);
              ValueType max_subcell_values = subcell_evaluator_old.get_subcell_value(0);
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
    Utilities::DistributedCellData<
      dim,
      std::pair<dealii::Tensor<1, n_components, number>, dealii::Tensor<1, n_components, number>>>
      min_max_subcell_values(
        matrix_free_context.mf.get_dof_handler(matrix_free_context.dof_idx).get_triangulation());
    matrix_free_context.mf.loop_cell_centric(compute_min_max_subcell_values,
                                             min_max_subcell_values,
                                             previous_solution);
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

        const dealii::VectorizedArray<number> tol(1e-5); // TODO: Hwo to deal with this
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

            ValueType min_boundary_subcell_value = face_evaluator.get_value(0);
            ValueType max_boundary_subcell_value = face_evaluator.get_value(0);
            for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
              {
                const std::array<dealii::types::boundary_id,
                                 dealii::VectorizedArray<number>::size()>
                           boundary_ids = matrix_free.get_faces_by_cells_boundary_id(cell, face);
                const bool is_at_boundary =
                  boundary_ids[0] != dealii::numbers::internal_face_boundary_id;
                if (!is_at_boundary)
                  continue;

                face_evaluator.reinit(cell, face);
                face_evaluator.gather_evaluate(current_solution, dealii::EvaluationFlags::values);

                min_boundary_subcell_value = face_evaluator.get_value(0);
                max_boundary_subcell_value = face_evaluator.get_value(0);

                for (const unsigned int q : face_evaluator.quadrature_point_indices())
                  {
                    const ValueType w = face_evaluator.get_value(q);

                    for (unsigned int c = 0; c < n_components; ++c)
                      {
                        min_boundary_subcell_value[c] =
                          dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                            w[c],
                            min_boundary_subcell_value[c],
                            w[c],
                            min_boundary_subcell_value[c]);
                        max_boundary_subcell_value[c] =
                          dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
                            w[c],
                            max_boundary_subcell_value[c],
                            w[c],
                            max_boundary_subcell_value[c]);
                      }
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
                                       min_boundary_subcell_value[c][lane]);
                            max_subcell_values[c][lane] =
                              std::max(max_subcell_values[c][lane],
                                       max_boundary_subcell_value[c][lane]);
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
} // namespace MeltPoolDG::HyperbolicPDETools
