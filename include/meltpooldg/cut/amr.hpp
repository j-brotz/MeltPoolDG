#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/amr_data.hpp>
#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>


namespace MeltPoolDG::CutUtil
{
  /**
   * same as AMR::refine_grid(), but adapted for cases where one or more DoFHandler objects are used
   * in CutFEM operations.
   *
   * @param distribute_level_set_dofs lambda function, that distributes the DoFs of the
   *                                  @param level_set_dof_handler . If no data is attached by
   *                                  @param attach_vectors, e.g., for AMR at the initial condition,
   *                                  this lambda function must also interpolate the initial level
   *                                  set solution onto the @param level_set_dof_vector
   *
   * same as in AMR::refine_grid():
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
  template <int dim, typename VectorType, typename number = VectorType::value_type>
  void
  refine_grid(const AMR::MarkCellsForRefinementType<dim>            &mark_cells_for_refinement,
              const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
              const std::function<void()>                           &post,
              const std::function<void()>                           &distribute_level_set_dofs,
              const dealii::DoFHandler<dim>                         &level_set_dof_handler,
              VectorType                                            &level_set_dof_vector,
              const std::function<void()>                           &setup_dof_system,
              const AdaptiveMeshingData<number>                     &amr,
              dealii::Triangulation<dim>                            &tria,
              const int                                              n_time_step);
} // namespace MeltPoolDG::CutUtil