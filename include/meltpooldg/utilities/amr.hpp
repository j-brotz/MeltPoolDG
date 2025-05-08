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
   */
  template <typename number>
  inline bool
  now(const AdaptiveMeshingData<number> &amr, const int n_time_step)
  {
    return not(n_time_step % amr.every_n_step) or n_time_step == 0;
  }

  /**
   * Refine the mesh in @param tria if the @param n_time_step is chosen to be refined by now().
   *
   * @param mark_cells_for_refinement lambda function of type MarkCellsForRefinementType, see its
   *                                  documentation for info
   * @param attach_vectors  lambda function of type AttachDoFHandlerAndVectorsType, that attaches
   *                        all DoFHandlers and their respective DoFVectors that ought to be
   *                        transferred to the new mesh
   * @param post this lambda function is run after AMR was executed
   * @param setup_dof_system setup the dof system, this includes:
   *                         - distribute DoFs on the new mesh
   *                         - create pratitioning for the new mesh
   *                         - setup constraints on the new mesh
   *                         - reinit the MatrixFree object for the new DoFs (ScratchData::build())
   *                         - initialize all DoF vectors for the new DoF
   */
  template <int dim, typename VectorType, typename number = typename VectorType::value_type>
  void
  refine_grid(const MarkCellsForRefinementType<dim>                 &mark_cells_for_refinement,
              const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
              const std::function<void()>                           &post,
              const std::function<void()>                           &setup_dof_system,
              const AdaptiveMeshingData<number>                     &amr,
              dealii::Triangulation<dim>                            &tria,
              const int                                              n_time_step);

  /**
   * Same as above, but for only one @param dof_handler
   */
  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const std::function<bool(dealii::Triangulation<dim> &)> &mark_cells_for_refinement,
              const std::function<void(std::vector<VectorType *> &)>  &attach_vectors,
              const std::function<void()>                             &post,
              const std::function<void()>                             &setup_dof_system,
              const AdaptiveMeshingData<number>                       &amr,
              dealii::DoFHandler<dim>                                 &dof_handler,
              const int                                                n_time_step);

  namespace internal
  {
    // Limit the maximum and minimum refinement levels of cells of the grid and do not
    // coarsen/refine cells along boundary
    // TODO use this function within mark_cells_for_refinement
    template <int dim, typename number>
    void
    limit_amr(dealii::Triangulation<dim> &tria, const AdaptiveMeshingData<number> &amr_data);
  } // namespace internal
} // namespace MeltPoolDG::AMR
