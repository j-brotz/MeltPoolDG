#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/solution_transfer.h>

#include <meltpooldg/utilities/amr_data.hpp>

#include <functional>
#include <memory>
#include <utility>
#include <vector>


namespace MeltPoolDG
{
  template <typename number>
  inline bool
  now(const AdaptiveMeshingData<number> &amr, const int n_time_step)
  {
    return not(n_time_step % amr.every_n_step) or n_time_step == 0;
  }

  template <int dim, typename VectorType, typename number = VectorType::value_type>
  void
  refine_grid(const std::function<bool(dealii::Triangulation<dim> &)> &mark_cells_for_refinement,
              const std::function<void(
                std::vector<std::pair<const dealii::DoFHandler<dim> *,
                                      std::function<void(std::vector<VectorType *> &)>>> &data)>
                                                &attach_vectors,
              const std::function<void()>       &post,
              const std::function<void()>       &setup_dof_system,
              const AdaptiveMeshingData<number> &amr,
              dealii::Triangulation<dim>        &tria,
              const int                          n_time_step)
  {
    if (not now(amr, n_time_step))
      return;

    if (mark_cells_for_refinement(tria) == false)
      return;

    std::vector<
      std::pair<const dealii::DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
      data;

    attach_vectors(data);

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    // Limit the maximum and minimum refinement levels of cells of the grid.
    if (tria.n_levels() > amr.max_grid_refinement_level)
      for (auto &cell : tria.active_cell_iterators_on_level(amr.max_grid_refinement_level))
        if (cell->is_locally_owned())
          cell->clear_refine_flag();
    if (amr.min_grid_refinement_level > 0)
      for (auto &cell : tria.active_cell_iterators_on_level(amr.min_grid_refinement_level))
        if (cell->is_locally_owned())
          cell->clear_coarsen_flag();

    // do not coarsen/refine cells along boundary
    if (amr.do_not_modify_boundary_cells)
      for (auto &cell : tria.active_cell_iterators())
        {
          if (not cell->is_locally_owned())
            continue;

          for (auto &face : cell->face_iterators())
            {
              if (not face->at_boundary())
                continue;
              if (cell->refine_flag_set())
                cell->clear_refine_flag();
              else
                cell->clear_coarsen_flag();
            }
        }

    // Initialize the triangulation change from the old grid to the new grid
    tria.prepare_coarsening_and_refinement();

    // Initialize the solution transfer from the old grid to the new grid
    std::vector<std::unique_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers(
      n_dof_handlers);

    std::vector<std::vector<VectorType *>>       new_grid_solutions(n_dof_handlers);
    std::vector<std::vector<const VectorType *>> old_grid_solutions(n_dof_handlers);
    std::vector<std::vector<bool>>               update_ghost_values(n_dof_handlers);

    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
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

    tria.execute_coarsening_and_refinement();

    // update dof-related scratch data to match the current triangulation
    setup_dof_system();

    // interpolate the given solution to the new discretization
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        for (const auto &v : new_grid_solutions[j])
          v->zero_out_ghost_values();

        solution_transfers[j]->interpolate(new_grid_solutions[j]);

        for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
          if (update_ghost_values[j][i])
            new_grid_solutions[j][i]->update_ghost_values();
      }

    post();
  }


  template <int dim, typename VectorType, typename number>
  void
  refine_grid(const std::function<bool(dealii::Triangulation<dim> &)> &mark_cells_for_refinement,
              const std::function<void(std::vector<VectorType *> &)>  &attach_vectors,
              const std::function<void()>                             &post,
              const std::function<void()>                             &setup_dof_system,
              const AdaptiveMeshingData<number>                       &amr,
              dealii::DoFHandler<dim>                                 &dof_handler,
              const int                                                n_time_step)
  {
    refine_grid<dim, VectorType>(
      mark_cells_for_refinement,
      [&](std::vector<std::pair<const dealii::DoFHandler<dim> *,
                                std::function<void(std::vector<VectorType *> &)>>> &data) {
        data.emplace_back(&dof_handler, attach_vectors);
      },
      post,
      setup_dof_system,
      amr,
      const_cast<dealii::Triangulation<dim> &>(dof_handler.get_triangulation()),
      n_time_step);
  }
} // namespace MeltPoolDG
