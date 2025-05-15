#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/solution_transfer.h>

#include <meltpooldg/utilities/attach_vectors.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>


namespace MeltPoolDG::CutUtil
{
  /**
   * same as Restart::deserialize_internal(), but adapted for cases where one or more DoFHandler
   * objects are used in CutFEM operations.
   *
   * Similarly to CutUtil::refine_grid(), the level set must be transferred first, so the CutFEM
   * operation can classify the mesh to distribute its DoFs before transferring its solution.
   *
   * @param distribute_level_set_dofs lambda function, that distributes the DoFs of the
   *                                  @param level_set_dof_handler .
   *
   * same as in Restart::deserialize_internal():
   * @param attach_vectors  lambda function of type AttachDoFHandlerAndVectorsType, that attaches
   *                        all DoFHandlers and their respective DoFVectors that ought to be
   *                        reconstructed
   * @param post this lambda function is run after AMR was executed
   * @param setup_dof_system setup the dof system, this includes:
   *                         - distribute DoFs on the new mesh
   *                         - create partitioning for the new mesh
   *                         - setup constraints on the new mesh
   *                         - reinit the MatrixFree object for the new DoFs (ScratchData::build())
   *                         - initialize all DoF vectors for the new DoF
   */
  template <int dim, typename VectorType>
  void
  deserialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                       const std::function<void()>                           &post,
                       const std::function<void()>   &distribute_level_set_dofs,
                       const dealii::DoFHandler<dim> &level_set_dof_handler,
                       const std::function<void()>   &setup_dof_system,
                       const std::string             &prefix)
  {
    DoFHandlerAndVectorDataType<dim, VectorType> data;
    attach_vectors(data);
    data.shrink_to_fit();

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    auto tria = dynamic_cast<dealii::parallel::distributed::Triangulation<dim> *>(
      const_cast<dealii::Triangulation<dim> *>(&data[0].first->get_triangulation()));
    AssertThrow(tria, dealii::ExcNotImplemented());

    tria->load(prefix + "_tria");

    // deserialize level set first, so the cut operation can classify the cells according to it

    // setup a temporary dof system for the level set
    distribute_level_set_dofs();

    // get pointers to all level set dof vectors from attach_vectors()
    std::vector<VectorType *> level_set_dof_vectors;
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        if (data[j].first != &level_set_dof_handler)
          continue;
        data[j].second(level_set_dof_vectors);
        break;
      }
    AssertThrow(level_set_dof_vectors.size() > 0,
                dealii::ExcMessage("The level set must be attached!"));

    //  initialize all level set dof vectors without MatrixFree
    {
      const auto &locally_owned_dofs = level_set_dof_handler.locally_owned_dofs();
      const auto  locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(level_set_dof_handler);
      for (auto vec : level_set_dof_vectors)
        vec->reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    level_set_dof_handler.get_communicator());
    }

    dealii::SolutionTransfer<dim, VectorType> ls_solution_transfer(level_set_dof_handler);
    ls_solution_transfer.deserialize(level_set_dof_vectors);

    // since the level set vectors were removed from the serialized data, we must make a copies
    std::vector<VectorType> level_set_vectors_buffer(level_set_dof_vectors.size());
    for (unsigned int i = 0; i < level_set_dof_vectors.size(); ++i)
      level_set_vectors_buffer[i] = *level_set_dof_vectors[i];

    // update dof-related scratch data including CutFEM operation that need the level set to do so
    setup_dof_system();

    for (unsigned int i = 0; i < level_set_dof_vectors.size(); ++i)
      {
        level_set_dof_vectors[i]->copy_locally_owned_data_from(level_set_vectors_buffer[i]);
        level_set_dof_vectors[i]->update_ghost_values();
      }

    // deserialize all remaining dof vectors
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        if (data[j].first == &level_set_dof_handler)
          continue;

        std::vector<VectorType *> new_grid_solutions;
        data[j].second(new_grid_solutions);

        dealii::SolutionTransfer<dim, VectorType> solution_transfer(*data[j].first);
        solution_transfer.deserialize(new_grid_solutions);
      }

    post();
  }
} // namespace MeltPoolDG::CutUtil