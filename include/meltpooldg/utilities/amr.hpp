#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>
#include <vector>


namespace MeltPoolDG::AMR
{
  /**
   * Type alias definition for the lambda function that determines how the mesh is to be refined.
   * If this function determines that the mesh does not change, it return false, true otherwise.
   */
  template <int dim>
  using MarkCellsForRefinementType = std::function<bool(dealii::Triangulation<dim> &)>;

  /**
   * determine whether the current @param n_time_step is chosen to execute AMR
   *
   * @param amr AMR parameters
   * @param n_time_step current time step number
   */
  template <typename number>
  inline bool
  now(const AdaptiveMeshingData<number> &amr, const int n_time_step)
  {
    return not(n_time_step % amr.every_n_step) or n_time_step == 0;
  }

  template <int dim, typename VectorType, typename number = typename VectorType::value_type>
  void
  refine_grid(const MarkCellsForRefinementType<dim>       &mark_cells_for_refinement,
              const DoFHandlerAndVectors<dim, VectorType> &dof_handler_and_vectors,
              const std::function<void()>                 &setup_dof_system,
              const AdaptiveMeshingData<number>           &amr,
              dealii::Triangulation<dim>                  &tria,
              const int                                    n_time_step,
              const std::function<void()>                 &post = std::function<void()>(),
              const std::function<void()>                 &pre  = std::function<void()>());

  /**
   * Refine the mesh in @param tria if the @param n_time_step is chosen to be refined by now().
   *
   * @param mark_cells_for_refinement lambda function of type MarkCellsForRefinementType, see its
   *                                  documentation for info
   * @param attach_vectors  lambda function of type AttachDoFHandlerAndVectorsType, that attaches
   *                        all DoFHandlers and their respective DoFVectors that ought to be
   *                        transferred to the new mesh
   * @param setup_dof_system setup the dof system, this includes:
   *                         - distribute DoFs on the new mesh
   *                         - create partitioning for the new mesh
   *                         - setup constraints on the new mesh
   *                         - reinit the MatrixFree object for the new DoFs (ScratchData::build())
   *                         - initialize all DoF vectors for the new DoF
   * @param post this optional lambda function is run after AMR was executed
   * @param pre this optional lambda function is run before AMR was executed
   *
   * @note This function is deprecated!
   */
  template <int dim, typename VectorType, typename number = typename VectorType::value_type>
  void
  refine_grid(const MarkCellsForRefinementType<dim>                 &mark_cells_for_refinement,
              const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
              const std::function<void()>                           &setup_dof_system,
              const AdaptiveMeshingData<number>                     &amr,
              dealii::Triangulation<dim>                            &tria,
              const int                                              n_time_step,
              const std::function<void()>                           &post = std::function<void()>(),
              const std::function<void()>                           &pre  = std::function<void()>())
  {
    DoFHandlerAndVectorDataType<dim, VectorType> dof_handler_and_vectors_to_refine_function;
    attach_vectors(dof_handler_and_vectors_to_refine_function);

    DoFHandlerAndVectors<dim, VectorType> dof_handler_and_vectors_to_refine;
    for (auto &elem : dof_handler_and_vectors_to_refine_function)
      {
        std::vector<VectorType *> temp;
        elem.second(temp);
        dof_handler_and_vectors_to_refine.emplace_back(elem.first, temp);
      }

    refine_grid(mark_cells_for_refinement,
                dof_handler_and_vectors_to_refine,
                setup_dof_system,
                amr,
                tria,
                n_time_step,
                post,
                pre);
  }

  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const std::function<bool(dealii::Triangulation<dim> &)> &mark_cells_for_refinement,
              dealii::Triangulation<dim>                              &tria,
              dealii::DoFHandler<dim>                                 &dof_handler,
              const std::vector<VectorType *>                         &dof_vectors,
              const std::function<void()>                             &setup_dof_system,
              const AdaptiveMeshingData<number>                       &amr,
              const int                                                n_time_step,
              const std::function<void()> &post = std::function<void()>(),
              const std::function<void()> &pre  = std::function<void()>());

  /**
   * Same as above, but for only one @param dof_handler (deprecated)
   */
  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const std::function<bool(dealii::Triangulation<dim> &)> &mark_cells_for_refinement,
              const std::function<void(std::vector<VectorType *> &)>  &attach_vectors,
              const std::function<void()>                             &setup_dof_system,
              const AdaptiveMeshingData<number>                       &amr,
              dealii::DoFHandler<dim>                                 &dof_handler,
              const int                                                n_time_step,
              const std::function<void()> &post = std::function<void()>(),
              const std::function<void()> &pre  = std::function<void()>())
  {
    std::vector<VectorType *> temp;
    attach_vectors(temp);
    refine_grid(mark_cells_for_refinement,
                const_cast<dealii::Triangulation<dim> &>(dof_handler.get_triangulation()),
                dof_handler,
                temp,
                setup_dof_system,
                amr,
                n_time_step,
                post,
                pre);
  }

  /**
   * Marks cells for refinement or coarsening based on the provided indicator.
   *
   * The marking strategy follows the approach described in deal.II’s documentation for
   * GridRefinement::refine_and_coarsen_fixed_number().
   *
   * @param tria Triangulation whose cells are to be marked for refinement or coarsening. Smaller
   * values indicate a lower need for refinement.
   * @param indicator Vector of refinement indicators for the active cells.
   * @param amr_data Object defining the lower and upper fractions of cells to refine or coarsen.
   * @param max_n_cells Maximum number of total cells allowed in the triangulation.
   */
  template <int dim, typename number>
  bool
  mark_fixed_number(dealii::Triangulation<dim>        &tria,
                    const dealii::Vector<number>      &indicator,
                    const AdaptiveMeshingData<number> &amr_data,
                    dealii::types::global_cell_index   max_n_cells =
                      std::numeric_limits<dealii::types::global_cell_index>::max());

  /**
   * Marks cells for refinement or coarsening based on the provided indicator.
   *
   * The marking strategy follows the approach described in deal.II’s documentation for
   * GridRefinement::refine_and_coarsen_fixed_fraction().
   *
   * @param tria Triangulation whose cells are to be marked for refinement or coarsening. Smaller
   * values indicate a lower need for refinement.
   * @param indicator  Vector of refinement indicators for the active cells.
   * @param amr_data Object defining the lower and upper fractions of cells to refine or coarsen.
   */
  template <int dim, typename number>
  bool
  mark_fixed_fraction(dealii::Triangulation<dim>        &tria,
                      const dealii::Vector<number>      &indicator,
                      const AdaptiveMeshingData<number> &amr_data);

  namespace internal
  {
    // Limit the maximum and minimum refinement levels of cells of the grid and do not
    // coarsen/refine cells along boundary
    // TODO use this function within mark_cells_for_refinement
    template <int dim, typename number>
    void
    limit_amr(dealii::Triangulation<dim> &tria, const AdaptiveMeshingData<number> &amr_data);

    /**
     * Clears refinement flags for cells whose indicators fall below a given threshold.
     *
     * The triangulation is assumed to already have refinement flags set. For each active cell
     * marked for refinement, the flag is removed if the corresponding indicator value is less
     * than the specified threshold.
     *
     * @param tria       Triangulation whose refinement flags may be modified.
     * @param indicator  Vector of refinement indicators for the active cells.
     * @param threshold  Minimum indicator value required to keep a cell marked for refinement.
     */
    template <int dim, typename number>
    void
    clear_refine_flags_below_threshold(dealii::Triangulation<dim>   &tria,
                                       const dealii::Vector<number> &indicator,
                                       number                        threshold);

    /**
     * Checks whether enough (global) cells in the triangulation are marked for refinement or
     * coarsening.
     *
     * Counts all locally owned active cells that have either the refinement or coarsening flag set.
     * The counts are summed across all MPI processes. Returns true if the total number of such
     * cells is greater than or equal to the given threshold.
     *
     * @param tria       Triangulation to inspect.
     * @param min_cells  Minimum number of flagged cells to return true.
     *
     * @return True if at least @p min_cells are marked for refinement or coarsening, false otherwise.
     */
    template <int dim>
    bool
    do_refine_based_on_n_marked_cells(const dealii::Triangulation<dim> &tria,
                                      unsigned                          min_cells = 1);
  } // namespace internal
} // namespace MeltPoolDG::AMR
