#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>

#include <meltpooldg/utilities/restart.hpp>

#include <memory>

namespace MeltPoolDG::Restart
{

  template <int dim, typename VectorType>
  void
  serialize_internal(
    const std::function<
      void(std::vector<std::pair<const DoFHandler<dim> *,
                                 std::function<void(std::vector<VectorType *> &)>>> &data)>
                      &attach_vectors,
    const std::string &prefix)
  {
    std::vector<
      std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
      data;
    attach_vectors(data);

    const unsigned int n = data.size();

    Assert(n > 0, ExcNotImplemented());

    auto triangulation = const_cast<Triangulation<dim> *>(&data[0].first->get_triangulation());

    Assert(triangulation, ExcNotImplemented());

    if (dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation))
      {
        auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation);

        std::vector<std::shared_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>>
          solution_transfer(n);

        std::vector<std::vector<VectorType *>>       new_grid_solutions(n);
        std::vector<std::vector<const VectorType *>> old_grid_solutions(n);

        for (unsigned int j = 0; j < n; ++j)
          {
            data[j].second(new_grid_solutions[j]);

            for (const auto &i : new_grid_solutions[j])
              {
                i->update_ghost_values();
                old_grid_solutions[j].push_back(i);
              }
            solution_transfer[j] =
              std::make_shared<parallel::distributed::SolutionTransfer<dim, VectorType>>(
                *data[j].first);
            solution_transfer[j]->prepare_for_serialization(old_grid_solutions[j]);
          }

        tria->save(prefix + "_tria");
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

  template <int dim, typename VectorType>
  void
  deserialize_internal(
    const std::function<
      void(std::vector<std::pair<const DoFHandler<dim> *,
                                 std::function<void(std::vector<VectorType *> &)>>> &data)>
                                &attach_vectors,
    const std::function<void()> &post,
    const std::function<void()> &setup_dof_system,
    const std::string           &prefix)
  {
    std::vector<
      std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
      data;
    attach_vectors(data);

    const unsigned int n = data.size();

    Assert(n > 0, ExcNotImplemented());

    auto triangulation = const_cast<Triangulation<dim> *>(&data[0].first->get_triangulation());

    Assert(triangulation, ExcNotImplemented());

    if (dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation))
      {
        auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation);

        tria->load(prefix + "_tria");

        setup_dof_system();

        std::vector<std::shared_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>>
          solution_transfer(n);

        std::vector<std::vector<VectorType *>> new_grid_solutions(n);

        for (unsigned int j = 0; j < n; ++j)
          {
            data[j].second(new_grid_solutions[j]);

            solution_transfer[j] =
              std::make_shared<parallel::distributed::SolutionTransfer<dim, VectorType>>(
                *data[j].first);
            solution_transfer[j]->deserialize(new_grid_solutions[j]);
          }

        post();
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }
} // namespace MeltPoolDG::Restart