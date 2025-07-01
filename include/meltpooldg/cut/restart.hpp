#pragma once

#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>
#include <string>


namespace MeltPoolDG::CutUtil
{
  /**
   * @brief Same as Restart::deserialize_internal(), but adapted for cases where one or more
   * DoFHandler objects are used in CutFEM operations.
   *
   * Similarly to CutUtil::refine_grid(), the level set must be transferred first, so the CutFEM
   * operation can classify the mesh to distribute its DoFs before transferring its solution.
   *
   * @param distribute_level_set_dofs Lambda function, that distributes the DoFs of the
   * @p level_set_dof_handler.
   * @param level_set_dof_handler DoF-Handler for the level-set field.
   *
   * same as in Restart::deserialize_internal():
   * @param attach_vectors  Lambda function of type AttachDoFHandlerAndVectorsType, that attaches
   *                        all DoFHandlers and their respective DoFVectors that ought to be
   *                        reconstructed.
   * @param post This lambda function is run after AMR was executed.
   * @param setup_dof_system Set up the dof system, this includes:
   *                         - distribute DoFs on the new mesh
   *                         - create partitioning for the new mesh
   *                         - setup constraints on the new mesh
   *                         - reinit the MatrixFree object for the new DoFs (ScratchData::build())
   *                         - initialize all DoF vectors for the new DoF layout
   * @param prefix File basename.
   */
  template <int dim, typename VectorType>
  void
  deserialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                       const std::function<void()>                           &post,
                       const std::function<void()>   &distribute_level_set_dofs,
                       const dealii::DoFHandler<dim> &level_set_dof_handler,
                       const std::function<void()>   &setup_dof_system,
                       const std::string             &prefix);
} // namespace MeltPoolDG::CutUtil
