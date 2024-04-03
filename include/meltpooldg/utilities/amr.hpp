/* ---------------------------------------------------------------------
 *
 * Author: Peter Munch, Magdalena Schreter, TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/solution_transfer.h>

namespace MeltPoolDG
{
  using namespace dealii;

  inline bool
  now(const AdaptiveMeshingData &amr, const int n_time_step)
  {
    return ((n_time_step == 0) || !(n_time_step % amr.every_n_step));
  }

  template <int dim, typename VectorType>
  void
  refine_grid(const std::function<bool(Triangulation<dim> &)> &mark_cells_for_refinement,
              const std::function<void(
                std::vector<std::pair<const DoFHandler<dim> *,
                                      std::function<void(std::vector<VectorType *> &)>>> &data)>
                                          &attach_vectors,
              const std::function<void()> &post,
              const std::function<void()> &setup_dof_system,
              const AdaptiveMeshingData   &amr,
              Triangulation<dim>          &tria,
              const int                    n_time_step)
  {
    if (!now(amr, n_time_step))
      return;

    if (mark_cells_for_refinement(tria) == false)
      return;

    std::vector<
      std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
      data;

    attach_vectors(data);

    const unsigned int n = data.size();

    Assert(n > 0, ExcNotImplemented());

    /*
     *  Limit the maximum and minimum refinement levels of cells of the grid.
     */
    if (tria.n_levels() > amr.max_grid_refinement_level)
      for (auto &cell : tria.active_cell_iterators_on_level(amr.max_grid_refinement_level))
        cell->clear_refine_flag();

    for (auto &cell : tria.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            if (cell->level() <= amr.min_grid_refinement_level)
              cell->clear_coarsen_flag();
            /*
             *  do not coarsen/refine cells along boundary
             */
            if (amr.do_not_modify_boundary_cells)
              {
                for (auto &face : cell->face_iterators())
                  if (face->at_boundary())
                    {
                      if (cell->refine_flag_set())
                        cell->clear_refine_flag();
                      else
                        cell->clear_coarsen_flag();
                    }
              }
          }
      }

    if (dynamic_cast<parallel::distributed::Triangulation<dim> *>(&tria))
      {
        /*
         *  Initialize the triangulation change from the old grid to the new grid
         */
        tria.prepare_coarsening_and_refinement();
        /*
         *  Initialize the solution transfer from the old grid to the new grid
         */
        std::vector<std::shared_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>>
          solution_transfer(n);

        std::vector<std::vector<VectorType *>>       new_grid_solutions(n);
        std::vector<std::vector<const VectorType *>> old_grid_solutions(n);
        std::vector<std::vector<bool>>               update_ghost_elements(n);

        for (unsigned int j = 0; j < n; ++j)
          {
            data[j].second(new_grid_solutions[j]);

            for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
              update_ghost_elements[j].push_back(new_grid_solutions[j][i]->has_ghost_elements());


            for (const auto &i : new_grid_solutions[j])
              {
                i->update_ghost_values();
                old_grid_solutions[j].push_back(i);
              }
            solution_transfer[j] =
              std::make_shared<parallel::distributed::SolutionTransfer<dim, VectorType>>(
                *data[j].first);
            solution_transfer[j]->prepare_for_coarsening_and_refinement(old_grid_solutions[j]);
          }
        /*
         *  Execute the grid refinement
         */
        tria.execute_coarsening_and_refinement();
        /*
         *  update dof-related scratch data to match the current triangulation
         */
        setup_dof_system();
        /*
         *  interpolate the given solution to the new discretization
         *
         */
        for (unsigned int j = 0; j < n; ++j)
          {
            for (const auto &i : new_grid_solutions[j])
              i->zero_out_ghost_values();
            solution_transfer[j]->interpolate(new_grid_solutions[j]);

            for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
              if (update_ghost_elements[j][i])
                new_grid_solutions[j][i]->update_ghost_values();
          }
        post();
      }
    else
      {
        /*
         *  Initialize the triangulation change from the old grid to the new grid
         */
        tria.prepare_coarsening_and_refinement();
        /*
         *  Initialize the solution transfer from the old grid to the new grid
         */
        std::vector<std::shared_ptr<SolutionTransfer<dim, VectorType>>> solution_transfer(n);

        std::vector<std::vector<VectorType *>>       new_grid_solutions(n);
        std::vector<std::vector<const VectorType *>> old_grid_solutions(n);
        std::vector<std::vector<bool>>               update_ghost_elements(n);

        std::vector<std::vector<VectorType>> new_grid_solutions_full(n);
        std::vector<std::vector<VectorType>> old_grid_solutions_full(n);

        for (unsigned int j = 0; j < n; ++j)
          {
            data[j].second(new_grid_solutions[j]);

            for (unsigned int i = 0; i < new_grid_solutions[j].size(); ++i)
              update_ghost_elements[j].push_back(new_grid_solutions[j][i]->has_ghost_elements());

            for (const auto &i : new_grid_solutions[j])
              {
                i->update_ghost_values();
                old_grid_solutions[j].push_back(i);
              }
            solution_transfer[j] =
              std::make_shared<SolutionTransfer<dim, VectorType>>(*data[j].first);

            old_grid_solutions_full[j].resize(new_grid_solutions[j].size());
            for (unsigned int i = 0; i < old_grid_solutions_full[j].size(); ++i)
              {
                const auto &distributed = *old_grid_solutions[j][i];
                IndexSet    ghost(distributed.size());
                ghost.add_range(0, distributed.size());
                old_grid_solutions_full[j][i].reinit(distributed.locally_owned_elements(),
                                                     ghost,
                                                     distributed.get_mpi_communicator());

                old_grid_solutions_full[j][i].copy_locally_owned_data_from(
                  *old_grid_solutions[j][i]);
                old_grid_solutions_full[j][i].update_ghost_values();
              }
            solution_transfer[j]->prepare_for_coarsening_and_refinement(old_grid_solutions_full[j]);
          }

        /*
         *  Execute the grid refinement
         */
        tria.execute_coarsening_and_refinement();
        /*
         *  update dof-related scratch data to match the current triangulation
         */
        setup_dof_system();
        /*
         *  interpolate the given solution to the new discretization
         */
        for (unsigned int j = 0; j < n; ++j)
          {
            new_grid_solutions_full[j].resize(new_grid_solutions[j].size());
            for (unsigned int i = 0; i < new_grid_solutions_full[j].size(); ++i)
              {
                const auto &distributed = *new_grid_solutions[j][i];
                IndexSet    ghost(distributed.size());
                ghost.add_range(0, distributed.size());
                new_grid_solutions_full[j][i].reinit(distributed.locally_owned_elements(),
                                                     ghost,
                                                     distributed.get_mpi_communicator());
              }

            solution_transfer[j]->interpolate(old_grid_solutions_full[j],
                                              new_grid_solutions_full[j]);

            for (unsigned int i = 0; i < new_grid_solutions_full[j].size(); ++i)
              {
                new_grid_solutions[j][i]->copy_locally_owned_data_from(
                  new_grid_solutions_full[j][i]);

                if (update_ghost_elements[j][i])
                  new_grid_solutions[j][i]->update_ghost_values();
              }
          }
        post();
      }
  }


  template <int dim, typename VectorType>
  void
  refine_grid(const std::function<bool(Triangulation<dim> &)>        &mark_cells_for_refinement,
              const std::function<void(std::vector<VectorType *> &)> &attach_vectors,
              const std::function<void()>                            &post,
              const std::function<void()>                            &setup_dof_system,
              const AdaptiveMeshingData                              &amr,
              DoFHandler<dim>                                        &dof_handler,
              const int                                               n_time_step)
  {
    refine_grid<dim, VectorType>(
      mark_cells_for_refinement,
      [&](std::vector<std::pair<const DoFHandler<dim> *,
                                std::function<void(std::vector<VectorType *> &)>>> &data) {
        data.emplace_back(&dof_handler, attach_vectors);
      },
      post,
      setup_dof_system,
      amr,
      const_cast<Triangulation<dim> &>(dof_handler.get_triangulation()),
      n_time_step);
  }

} // namespace MeltPoolDG
