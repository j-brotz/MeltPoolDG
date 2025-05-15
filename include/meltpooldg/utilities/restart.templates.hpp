#pragma once

#include <meltpooldg/utilities/restart.hpp>
//
#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/solution_transfer.h>

#include <memory>

namespace MeltPoolDG::Restart
{
  template <int dim, typename VectorType>
  void
  serialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                     const std::string                                     &prefix)
  {
    DoFHandlerAndVectorDataType<dim, VectorType> data;
    attach_vectors(data);
    data.shrink_to_fit();

    const unsigned int n_dof_handlers = data.size();

    Assert(n_dof_handlers > 0, dealii::ExcNotImplemented());

    auto tria = dynamic_cast<dealii::parallel::distributed::Triangulation<dim> *>(
      const_cast<dealii::Triangulation<dim> *>(&data[0].first->get_triangulation()));
    AssertThrow(tria, dealii::ExcNotImplemented());

    // the SolutionTransfer instances may not be destructed until Triangulation::save() is called
    std::vector<std::shared_ptr<dealii::SolutionTransfer<dim, VectorType>>> solution_transfers(
      n_dof_handlers);
    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        std::vector<VectorType *>       new_grid_solutions;
        std::vector<const VectorType *> old_grid_solutions;
        data[j].second(new_grid_solutions);

        for (const auto &i : new_grid_solutions)
          {
            i->update_ghost_values();
            old_grid_solutions.push_back(i);
          }

        solution_transfers[j] =
          std::make_unique<dealii::SolutionTransfer<dim, VectorType>>(*data[j].first);
        solution_transfers[j]->prepare_for_serialization(old_grid_solutions);
      }

    tria->save(prefix + "_tria");
  }

  template <int dim, typename VectorType>
  void
  deserialize_internal(const AttachDoFHandlerAndVectorsType<dim, VectorType> &attach_vectors,
                       const std::function<void()>                           &post,
                       const std::function<void()>                           &setup_dof_system,
                       const std::string                                     &prefix)
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

    setup_dof_system();

    for (unsigned int j = 0; j < n_dof_handlers; ++j)
      {
        std::vector<VectorType *> new_grid_solutions;
        data[j].second(new_grid_solutions);

        dealii::SolutionTransfer<dim, VectorType> solution_transfer(*data[j].first);
        solution_transfer.deserialize(new_grid_solutions);
      }

    post();
  }
} // namespace MeltPoolDG::Restart
