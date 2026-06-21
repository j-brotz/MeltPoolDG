#include <meltpooldg/utilities/triangulation_utils.hpp>
//
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <map>
#include <utility>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, int spacedim>
  TriangulationType
  get_triangulation_type(const dealii::Triangulation<dim, spacedim> &tria)
  {
    if (dynamic_cast<dealii::parallel::shared::Triangulation<dim, spacedim> *>(
          const_cast<dealii::Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::shared;
    else if (dynamic_cast<dealii::parallel::distributed::Triangulation<dim, spacedim> *>(
               const_cast<dealii::Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::parallel_distributed;
    else if (dynamic_cast<dealii::parallel::fullydistributed::Triangulation<dim, spacedim> *>(
               const_cast<dealii::Triangulation<dim, spacedim> *>(&tria)))
      return TriangulationType::parallel_fullydistributed;
    else
      return TriangulationType::serial;
  }

  template <int dim>
  LevelCommunicationPattern<dim>::LevelCommunicationPattern(
    const dealii::parallel::distributed::Triangulation<dim> &tria)
    : triangulation(tria)
    , mpi_communicator(triangulation.get_mpi_communicator())
  {}

  template <int dim>
  void
  LevelCommunicationPattern<dim>::build_pattern(const unsigned int level)
  {
    Assert(
      triangulation.n_global_levels() > 1,
      dealii::ExcMessage(
        "The triangulation must have at least two levels to build a level specific communication pattern."));

    Assert(level < triangulation.n_global_levels(),
           dealii::ExcMessage(
             std::string("The specified level for which the communication pattern is to be built "
                         "exceeds the maximum level of the triangulation.")));

    partition_level = level;
    build_communication_pattern();
  }

  template <int dim>
  void
  LevelCommunicationPattern<dim>::assert_pattern_built() const
  {
    Assert(
      is_pattern_built,
      dealii::ExcMessage(
        "The communication pattern has not been built yet. Please call build_pattern() first."));
  }

  template <int dim>
  std::map<unsigned int, std::vector<dealii::CellId>>
  LevelCommunicationPattern<dim>::cells_to_send() const
  {
    assert_pattern_built();
    return rank_to_cells_send;
  }

  template <int dim>
  std::map<unsigned int, std::vector<dealii::CellId>>
  LevelCommunicationPattern<dim>::cells_to_receive() const
  {
    assert_pattern_built();
    return rank_to_cells_receive;
  }

  template <int dim>
  unsigned int
  LevelCommunicationPattern<dim>::n_send_ranks() const
  {
    assert_pattern_built();
    return rank_to_cells_send.size();
  }

  template <int dim>
  unsigned int
  LevelCommunicationPattern<dim>::n_receive_ranks() const
  {
    assert_pattern_built();
    return rank_to_cells_receive.size();
  }

  template <int dim>
  std::vector<unsigned int>
  LevelCommunicationPattern<dim>::send_ranks() const
  {
    assert_pattern_built();
    std::vector<unsigned int> ranks;
    ranks.reserve(rank_to_cells_send.size());

    for (const auto &rank : std::views::keys(rank_to_cells_send))
      ranks.push_back(rank);

    return ranks;
  }

  template <int dim>
  std::vector<unsigned int>
  LevelCommunicationPattern<dim>::receive_ranks() const
  {
    assert_pattern_built();
    std::vector<unsigned int> ranks;
    ranks.reserve(rank_to_cells_receive.size());

    for (const auto &rank : std::views::keys(rank_to_cells_receive))
      ranks.push_back(rank);

    return ranks;
  }

  template <int dim>
  std::vector<dealii::CellId>
  LevelCommunicationPattern<dim>::owned_active_cells_ancestors() const
  {
    // lambda that determines for a given cell and a given level the parent cell on that level
    const auto level_parent_cell_id = [](const auto                       level_parent_cell_id,
                                         const dealii::CellAccessor<dim> &cell,
                                         int                              level) -> dealii::CellId {
      if (cell.level() == level)
        return cell.id();
      else
        return level_parent_cell_id(level_parent_cell_id, *cell.parent(), level);
    };

    std::set<dealii::CellId> parent_cells;
    for (auto &cell : triangulation.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          parent_cells.insert(level_parent_cell_id(level_parent_cell_id, *cell, partition_level));
      }

    return {std::vector<dealii::CellId>(parent_cells.begin(), parent_cells.end())};
  }

  template <int dim>
  std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
  LevelCommunicationPattern<dim>::gather_ancestors(
    const std::vector<dealii::CellId> &local_ancestor_cells) const
  {
    std::vector<std::pair<unsigned int, std::vector<dealii::CellId>>> all_parent_cells =
      dealii::Utilities::MPI::all_gather(mpi_communicator,
                                         std::make_pair(dealii::Utilities::MPI::this_mpi_process(
                                                          mpi_communicator),
                                                        local_ancestor_cells));

    return all_parent_cells;
  }

  template <int dim>
  std::vector<dealii::CellId>
  LevelCommunicationPattern<dim>::adjacent_relevant_cells(
    const dealii::CellId              &cell_id,
    const std::vector<dealii::CellId> &owned_active_cell_ancestors) const
  {
    Assert(triangulation.contains_cell(cell_id),
           dealii::ExcMessage(
             "The triangulation does not contain the cell with the given cell id."));

    std::vector<dealii::CellId> relevant_cells;

    typename dealii::Triangulation<dim>::cell_iterator cell =
      triangulation.create_cell_iterator(cell_id);

    // If the cell contains a locally owned active cell as descendant, the cell itself is relevant
    if (Utils::contains(owned_active_cell_ancestors, cell_id))
      {
        relevant_cells.push_back(cell_id);
      }


    // To determine all adjacent cells, we recursively traverse neighboring cells up to a depth of
    // `dim`. As the mesh data structure provides direct access only to face neighbors, cells that
    // are adjacent exclusively through a vertex are not immediately accessible. To capture these
    // cases, we recursively examine the face neighbors of already discovered neighbors, restricting
    // the search to faces that share a vertex with the cell under consideration. This ensures that
    // every vertex-connected cell is found.
    // TODO: What if the neighbor is not on the same level but on one level coarser?
    auto search_neighbors =
      [&relevant_cells,
       &owned_active_cell_ancestors](const auto &search_neighbors,
                                     const typename dealii::Triangulation<dim>::cell_iterator &cell,
                                     std::vector<unsigned> excluded_indices = {},
                                     int                   current_depth    = 1) -> void {
      for (unsigned int neighbor_index = 0;
           neighbor_index < dealii::GeometryInfo<dim>::faces_per_cell;
           ++neighbor_index)
        {
          if (std::ranges::find(excluded_indices, neighbor_index) == excluded_indices.end() and
              not cell->at_boundary(neighbor_index))
            {
              // TODO: What if the neighbor is not on the same level but on one level coarser?
              Assert(cell->neighbor(neighbor_index)->level() == cell->level(),
                     dealii::ExcMessage(std::string(
                       "The communication pattern for cells on a given level can currently only be "
                       "built if all cells on that level have neighbors on the same level, this "
                       "includes artificial cells which are owned by another rank.")));

              if (Utils::contains(owned_active_cell_ancestors,
                                  cell->neighbor(neighbor_index)->id()))
                relevant_cells.push_back(cell->neighbor(neighbor_index)->id());
              if (current_depth < dim)
                {
                  excluded_indices.push_back(neighbor_index);
                  search_neighbors(search_neighbors,
                                   cell->neighbor(neighbor_index),
                                   excluded_indices,
                                   current_depth + 1);
                }
            }
        }
    };

    search_neighbors(search_neighbors, cell);

    std::ranges::sort(relevant_cells, [](const auto &a, const auto &b) { return a < b; });
    relevant_cells.erase(std::ranges::unique(relevant_cells).begin(), relevant_cells.end());

    return relevant_cells;
  }

  template <int dim>
  std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
  LevelCommunicationPattern<dim>::relevant_cells_for_ranks(
    const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &requested_cells_by_rank,
    const std::vector<dealii::CellId> &owned_active_cell_ancestors) const
  {
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
      locally_owned_relevant_cells_for_ranks;

    for (const auto &[rank, cells] : requested_cells_by_rank)
      {
        if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
          {
            std::set<dealii::CellId> locally_owned_relevant_cells_for_rank;
            for (const auto &cell_id : cells)
              {
                // TODO: What if the cell is not available but the parent cell is available?
                if (triangulation.contains_cell(cell_id))
                  {
                    std::vector<dealii::CellId> relevant_cells =
                      adjacent_relevant_cells(cell_id, owned_active_cell_ancestors);
                    locally_owned_relevant_cells_for_rank.insert(relevant_cells.begin(),
                                                                 relevant_cells.end());
                  }
              }
            if (not locally_owned_relevant_cells_for_rank.empty())
              locally_owned_relevant_cells_for_ranks.emplace_back(
                rank,
                std::vector<dealii::CellId>(locally_owned_relevant_cells_for_rank.begin(),
                                            locally_owned_relevant_cells_for_rank.end()));
          }
      }

    return locally_owned_relevant_cells_for_ranks;
  }

  template <int dim>
  std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
  LevelCommunicationPattern<dim>::exchange_relevant_cells(
    const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &cells_to_send) const
  {
    // determine ranks to receive data from
    std::vector<unsigned int> destination_ranks;
    destination_ranks.reserve(cells_to_send.size());
    for (const auto &rank_and_cells : cells_to_send)
      {
        destination_ranks.push_back(rank_and_cells.first);
      }

    std::vector<unsigned int> source_ranks =
      dealii::Utilities::MPI::compute_point_to_point_communication_pattern(mpi_communicator,
                                                                           destination_ranks);

    // send the cell data to the corresponding ranks
    std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
    send_futures.reserve(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) - 1);
    for (const auto &[rank, cells] : cells_to_send)
      {
        if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
          {
            send_futures.push_back(dealii::Utilities::MPI::isend(
              std::make_pair(dealii::Utilities::MPI::this_mpi_process(mpi_communicator), cells),
              mpi_communicator,
              rank,
              0));
          }
      }

    // receive cell data from the corresponding ranks
    std::vector<dealii::Utilities::MPI::Future<std::pair<unsigned, std::vector<dealii::CellId>>>>
      receive_futures;
    for (unsigned int rank : source_ranks)
      {
        Assert(rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator),
               dealii::ExcMessage("The current process should not receive data from itself."));

        receive_futures.push_back(
          dealii::Utilities::MPI::irecv<std::pair<unsigned, std::vector<dealii::CellId>>>(
            mpi_communicator, rank, 0));
      }


    // wait for all communication to finish and gather the received data
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> received_data;
    received_data.reserve(receive_futures.size());
    for (auto &future : receive_futures)
      {
        future.wait();
        received_data.push_back(future.get());
      }
    for (auto &future : send_futures)
      future.wait();

    return received_data;
  }

  template <int dim>
  void
  LevelCommunicationPattern<dim>::store_rank_cell_maps(
    const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &data_to_receive,
    const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &data_to_send)
  {
    for (const auto &[rank, cells] : data_to_receive)
      {
        if (not cells.empty())
          rank_to_cells_receive[rank].insert(rank_to_cells_receive[rank].end(),
                                             cells.begin(),
                                             cells.end());
      }

    for (const auto &[rank, cells] : data_to_send)
      {
        if (not cells.empty())
          {
            rank_to_cells_send[rank].insert(rank_to_cells_send[rank].end(),
                                            cells.begin(),
                                            cells.end());
          }
      }
  }

  template <int dim>
  void
  LevelCommunicationPattern<dim>::build_communication_pattern()
  {
    rank_to_cells_send.clear();
    rank_to_cells_receive.clear();

    // Cells whose active descendants are locally owned
    std::vector<dealii::CellId> owned_active_ancestors = owned_active_cells_ancestors();

    // Gather ancestor cells from all ranks
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> ancestor_cells_by_rank =
      gather_ancestors(owned_active_ancestors);

    // Determine which local cells are relevant for other ranks
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> cells_to_send =
      relevant_cells_for_ranks(ancestor_cells_by_rank, owned_active_ancestors);

    // Exchange relevant cells
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> cells_to_receive =
      exchange_relevant_cells(cells_to_send);

    // Store final communication pattern
    store_rank_cell_maps(cells_to_receive, cells_to_send);

    is_pattern_built = true;
  }


  template TriangulationType
  get_triangulation_type(const dealii::Triangulation<1, 1> &);
  template TriangulationType
  get_triangulation_type(const dealii::Triangulation<2, 2> &);
  template TriangulationType
  get_triangulation_type(const dealii::Triangulation<3, 3> &);

  template class LevelCommunicationPattern<1>;
  template class LevelCommunicationPattern<2>;
  template class LevelCommunicationPattern<3>;

} // namespace MeltPoolDG
