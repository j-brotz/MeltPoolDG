#include <meltpooldg/cut/amr.hpp>
//
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/solution_transfer.h>

#include <memory>
#include <vector>


namespace MeltPoolDG::CutUtil
{
  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const AMR::MarkCellsForRefinementType<dim>                 &mark_cells_for_refinement,
              const AMR::AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
              const std::function<void()>                                &post,
              const std::function<void()>                                &distribute_level_set_dofs,
              const dealii::DoFHandler<dim> *const                        level_set_dof_handler,
              const std::function<void()>       &setup_remaining_dof_system,
              const AdaptiveMeshingData<number> &amr,
              dealii::Triangulation<dim>        &tria,
              const int                          n_time_step)
  {
    if (not AMR::now(amr, n_time_step))
      return;

    if (mark_cells_for_refinement(tria) == false)
      return;

    AMR::internal::limit_amr(tria, amr);

    // the following is very similar to the code in AMR::refine_grid() and
    // CutUtil::SolutionTransferOperator::transfer_solution_constant_dofs()

    if (not attach_vectors)
      {
        // short circuit this function
        tria.prepare_coarsening_and_refinement();
        tria.execute_coarsening_and_refinement();
        distribute_level_set_dofs();
        setup_remaining_dof_system();
        post();
        return;
      }

    AMR::DoFHandlerAndVectorDataType<dim, VectorType> data;

    attach_vectors(data);

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    // Initialize the triangulation change from the old grid to the new grid
    tria.prepare_coarsening_and_refinement();

    // Initialize the solution transfer from the old grid to the new grid
    std::vector<std::unique_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers(
      n_dof_handlers);

    std::vector<std::vector<VectorType *>>       new_grid_solutions(n_dof_handlers);
    std::vector<std::vector<const VectorType *>> old_grid_solutions(n_dof_handlers);
    std::vector<std::vector<bool>>               update_ghost_values(n_dof_handlers);

    unsigned int ls_idx = dealii::numbers::invalid_unsigned_int;

    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        if (data[j].first == level_set_dof_handler)
          ls_idx = j;

        // collect pointers to the DoF vectors for the current DoFHandler
        data[j].second(new_grid_solutions[j]);

        old_grid_solutions[j].resize(new_grid_solutions[j].size());
        update_ghost_values[j].resize(new_grid_solutions[j].size(), true);
        for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
          {
            old_grid_solutions[j][i] = new_grid_solutions[j][i];
            if (not new_grid_solutions[j][i]->has_ghost_elements())
              {
                update_ghost_values[j][i] = false;
                new_grid_solutions[j][i]->update_ghost_values();
              }
          }

        solution_transfers[j] =
          std::make_unique<dealii::SolutionTransfer<dim, VectorType>>(*data[j].first);
        solution_transfers[j]->prepare_for_coarsening_and_refinement(old_grid_solutions[j]);
      }
    AssertThrow(ls_idx != dealii::numbers::invalid_unsigned_int,
                dealii::ExcMessage("The level set DoFHandler must be attached by attach_vectors!"));

    tria.execute_coarsening_and_refinement();

    distribute_level_set_dofs();

    {
      for (const auto &v : new_grid_solutions[ls_idx])
        v->zero_out_ghost_values();

      solution_transfers[ls_idx]->interpolate(new_grid_solutions[ls_idx]);

      for (unsigned int i = 0; i < new_grid_solutions[ls_idx].size(); ++i)
        if (update_ghost_values[ls_idx][i])
          new_grid_solutions[ls_idx][i]->update_ghost_values();
    }

    // update dof-related scratch data to match the current triangulation
    setup_remaining_dof_system();

    // interpolate the given solution to the new discretization
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        if (j == ls_idx)
          continue;

        for (const auto &v : new_grid_solutions[j])
          v->zero_out_ghost_values();

        solution_transfers[j]->interpolate(new_grid_solutions[j]);

        for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
          if (update_ghost_values[j][i])
            new_grid_solutions[j][i]->update_ghost_values();
      }

    post();
  }

  template void
  refine_grid(
    const AMR::MarkCellsForRefinementType<1> &,
    const AMR::AttachDoFHandlerAndVectorsType<1, dealii::LinearAlgebra::distributed::Vector<double>>
      &,
    const std::function<void()> &,
    const std::function<void()> &,
    const dealii::DoFHandler<1> *,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::Triangulation<1> &,
    const int);
  template void
  refine_grid(
    const AMR::MarkCellsForRefinementType<2> &,
    const AMR::AttachDoFHandlerAndVectorsType<2, dealii::LinearAlgebra::distributed::Vector<double>>
      &,
    const std::function<void()> &,
    const std::function<void()> &,
    const dealii::DoFHandler<2> *,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::Triangulation<2> &,
    const int);
  template void
  refine_grid(
    const AMR::MarkCellsForRefinementType<3> &,
    const AMR::AttachDoFHandlerAndVectorsType<3, dealii::LinearAlgebra::distributed::Vector<double>>
      &,
    const std::function<void()> &,
    const std::function<void()> &,
    const dealii::DoFHandler<3> *,
    const std::function<void()> &,
    const AdaptiveMeshingData<double> &,
    dealii::Triangulation<3> &,
    const int);
} // namespace MeltPoolDG::CutUtil
