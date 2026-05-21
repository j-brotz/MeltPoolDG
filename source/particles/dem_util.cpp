#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <meltpooldg/particles/dem_util.hpp>

#include <algorithm>
#include <map>
#include <ranges>
#include <utility>
#include <vector>

template <int dim>
MeltPoolDG::LevelCellCommunicationPattern<dim>::LevelCellCommunicationPattern(
  const dealii::Triangulation<dim> &tria)
  : triangulation(tria)
  , mpi_communicator(triangulation.get_mpi_communicator())
{}

template <int dim>
void
MeltPoolDG::LevelCellCommunicationPattern<dim>::build_pattern(const unsigned int level)
{
  partition_level = level;
  build_communication_pattern();
}

template <int dim>
std::map<unsigned int, std::vector<dealii::CellId>>
MeltPoolDG::LevelCellCommunicationPattern<dim>::cells_to_send() const
{
  return rank_to_cells_send;
}

template <int dim>
std::map<unsigned int, std::vector<dealii::CellId>>
MeltPoolDG::LevelCellCommunicationPattern<dim>::cells_to_receive() const
{
  return rank_to_cells_receive;
}

template <int dim>
unsigned int
MeltPoolDG::LevelCellCommunicationPattern<dim>::n_send_ranks() const
{
  return rank_to_cells_send.size();
}

template <int dim>
unsigned int
MeltPoolDG::LevelCellCommunicationPattern<dim>::n_receive_ranks() const
{
  return rank_to_cells_receive.size();
}

template <int dim>
std::vector<unsigned int>
MeltPoolDG::LevelCellCommunicationPattern<dim>::send_ranks() const
{
  std::vector<unsigned int> ranks;
  ranks.reserve(rank_to_cells_send.size());

  for (const auto &rank : std::views::keys(rank_to_cells_send))
    ranks.push_back(rank);

  return ranks;
}

template <int dim>
std::vector<unsigned int>
MeltPoolDG::LevelCellCommunicationPattern<dim>::receive_ranks() const
{
  std::vector<unsigned int> ranks;
  ranks.reserve(rank_to_cells_receive.size());

  for (const auto &rank : std::views::keys(rank_to_cells_receive))
    ranks.push_back(rank);

  return ranks;
}

template <int dim>
std::vector<dealii::CellId>
MeltPoolDG::LevelCellCommunicationPattern<dim>::owned_active_cells_ancestors() const
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
MeltPoolDG::LevelCellCommunicationPattern<dim>::gather_ancestors(
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
MeltPoolDG::LevelCellCommunicationPattern<dim>::adjacent_relevant_cells(
  const dealii::CellId              &cell_id,
  const std::vector<dealii::CellId> &owned_active_cell_ancestors) const
{
  std::vector<dealii::CellId> relevant_cells;

  typename dealii::Triangulation<dim>::cell_iterator cell =
    triangulation.create_cell_iterator(cell_id);

  // If the cell contains a locally owned active cell as descendant, the cell itself is relevant
  if (Utils::contains(owned_active_cell_ancestors, cell_id))
    {
      relevant_cells.push_back(cell_id);
    }

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
            if (Utils::contains(owned_active_cell_ancestors, cell->neighbor(neighbor_index)->id()))
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
MeltPoolDG::LevelCellCommunicationPattern<dim>::relevant_cells_for_ranks(
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
              std::vector<dealii::CellId> relevant_cells =
                adjacent_relevant_cells(cell_id, owned_active_cell_ancestors);
              locally_owned_relevant_cells_for_rank.insert(relevant_cells.begin(),
                                                           relevant_cells.end());
            }
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
MeltPoolDG::LevelCellCommunicationPattern<dim>::exchange_relevant_cells(
  const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &cells_to_send) const
{
  // send
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

  // receive
  std::vector<dealii::Utilities::MPI::Future<std::pair<unsigned, std::vector<dealii::CellId>>>>
    receive_futures;
  for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
       ++rank)
    {
      if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
        {
          receive_futures.push_back(
            dealii::Utilities::MPI::irecv<std::pair<unsigned, std::vector<dealii::CellId>>>(
              mpi_communicator, rank, 0));
        }
    }


  // wait
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
MeltPoolDG::LevelCellCommunicationPattern<dim>::store_rank_cell_maps(
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
MeltPoolDG::LevelCellCommunicationPattern<dim>::build_communication_pattern()
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

  // Exchange relevance information
  std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> cells_to_receive =
    exchange_relevant_cells(cells_to_send);

  // Store final communication pattern
  store_rank_cell_maps(cells_to_receive, cells_to_send);
}

template class MeltPoolDG::LevelCellCommunicationPattern<1>;
template class MeltPoolDG::LevelCellCommunicationPattern<2>;
template class MeltPoolDG::LevelCellCommunicationPattern<3>;
