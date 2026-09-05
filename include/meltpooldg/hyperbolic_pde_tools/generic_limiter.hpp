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

#include <meltpooldg/hyperbolic_pde_tools/fe_subcell_evaluation.hpp>
#include <meltpooldg/hyperbolic_pde_tools/fe_subcell_evaluation.templates.hpp>
#include <meltpooldg/hyperbolic_pde_tools/limiter_data.hpp>
#include <meltpooldg/hyperbolic_pde_tools/troubled_cell_indicators.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/dealii_tensor.hpp>
#include <meltpooldg/utilities/matrix_free_util.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <functional>
#include <iostream>
#include <utility>

namespace MeltPoolDG::HyperbolicPDETools
{
  // TODO: This assumes a cartesian grid with NO local mesh refinement
  template <int dim, int n_components, typename number>
  class Limiter
  {
    using VectorType          = dealii::LinearAlgebra::distributed::Vector<number>;
    using VectorizedArrayType = dealii::VectorizedArray<number>;
    using ValueType           = dealii::Tensor<1, n_components, VectorizedArrayType>;
    using FluxType            = dealii::Tensor<1, n_components, VectorizedArrayType>;

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
      const std::function<FluxType(
        const ValueType                                               &w_m,
        const ValueType                                               &w_p,
        const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal)> numerical_flux,
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
      cells_marked += inter_cell_numerical_admissibility_troubled_cell_marking(
        matrix_free_context, previous_time_solution, solution, troubled_cells, get_boundary_value);

    if (Utils::contains(limiter_data.troubled_cell_marking_types,
                        TroubledCellMarkingType::local_cell_numerical_admissibility))
      cells_marked += local_cell_numerical_admissibility_troubled_cell_marking(
        matrix_free_context, previous_time_solution, solution, troubled_cells, get_boundary_value);

    if (Utils::contains(limiter_data.troubled_cell_marking_types,
                        TroubledCellMarkingType::physical_admissibility))
      cells_marked += physical_admissibility_troubled_cell_marking(matrix_free_context,
                                                                   solution,
                                                                   troubled_cells,
                                                                   admissibility_check);

    return cells_marked;
  }

  template <int dim, int n_components, typename number>
  unsigned int
  Limiter<dim, n_components, number>::apply_limiting(
    const number                                                              time_step,
    const std::function<FluxType(
      const ValueType                                               &w_m,
      const ValueType                                               &w_p,
      const dealii::Tensor<1, dim, dealii::VectorizedArray<number>> &normal)> numerical_flux,
    std::function<ValueType(const dealii::Point<dim, dealii::VectorizedArray<number>> &,
                            dealii::types::boundary_id,
                            const ValueType &)>                               get_boundary_value,
    VectorType                                                               &limited_solution,
    const VectorType                                                         &solution,
    const std::function<dealii::VectorizedArray<number>(const ValueType &)>  &admissibility_check)
  {
    limited_solution = solution;
    if (limited_solution.has_ghost_elements())
      limited_solution.zero_out_ghost_values();

    const unsigned int n_cells_to_limit =
      mark_cells_for_limiting(solution, get_boundary_value, admissibility_check);

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
                    const ValueType neighbor =
                      subcell_evaluator.get_subcell_neighbor_value(subcell, subcell_face);
                    const dealii::Tensor<1, dim, VectorizedArrayType> normal =
                      subcell_evaluator.subcell_face_normal(subcell, subcell_face);
                    const FluxType flux = numerical_flux(w_old, neighbor, normal);

                    for (unsigned int c = 0; c < n_components; ++c)
                      fv_subcell_average[c] -=
                        prefactor * subcell_evaluator.subcell_face_size(subcell, subcell_face) *
                        flux[c];
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
} // namespace MeltPoolDG::HyperbolicPDETools
