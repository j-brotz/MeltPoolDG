#include <gtest/gtest.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/particles/dem_util.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

constexpr int dim = 2;
using number      = double;

class ParticleDataStructureTest : public ::testing::Test
{
protected:
  ParticleDataStructureTest()
    : triangulation(
        MPI_COMM_WORLD,
        dealii::Triangulation<dim>::MeshSmoothing::none,
        dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy)
  {
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(4);
  }
  dealii::parallel::distributed::Triangulation<dim> triangulation;
};

TEST_F(ParticleDataStructureTest, PatternReceiverRanks)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  constexpr int level_to_store_particles = 1;

  MeltPoolDG::LevelCellCommunicationPattern<dim> partitioner(triangulation);
  partitioner.build_pattern(level_to_store_particles);

  {
    SCOPED_TRACE("Check for correct receiver of data from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));

    // These are the owner ranks of the cells on level 1.
    std::vector<unsigned int> reference_receiver_ranks(
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    std::iota(reference_receiver_ranks.begin(), reference_receiver_ranks.end(), 0);

    auto range = std::ranges::remove(reference_receiver_ranks,
                                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    reference_receiver_ranks.erase(range.begin(), range.end());

    std::vector<unsigned int> partitioner_receiver_ranks = partitioner.receive_ranks();
    std::ranges::sort(partitioner_receiver_ranks);

    EXPECT_EQ(reference_receiver_ranks, partitioner_receiver_ranks);
  }
}

TEST_F(ParticleDataStructureTest, PatternSenderRanks)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  constexpr int level_to_store_particles = 1;

  MeltPoolDG::LevelCellCommunicationPattern<dim> partitioner(triangulation);
  partitioner.build_pattern(level_to_store_particles);

  {
    SCOPED_TRACE("Check for correct sender of data from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));

    std::vector<unsigned int> reference_sender_ranks;
    reference_sender_ranks.resize(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    std::iota(reference_sender_ranks.begin(), reference_sender_ranks.end(), 0);
    auto range = std::ranges::remove(reference_sender_ranks,
                                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    reference_sender_ranks.erase(range.begin(), range.end());

    std::vector<unsigned int> partitioner_sender_ranks = partitioner.send_ranks();
    std::ranges::sort(partitioner_sender_ranks);

    EXPECT_EQ(reference_sender_ranks, partitioner_sender_ranks);
  }

  {
    SCOPED_TRACE("Check that correct cells are sent from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));
    std::map<unsigned int, std::vector<dealii::CellId>> partitioner_cell_to_rank_send =
      partitioner.cells_to_send();


    std::map<unsigned int, std::vector<dealii::CellId>> reference_cell_to_rank_send;
    for (unsigned rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++rank)
      {
        if (rank != dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
          {
            reference_cell_to_rank_send[rank] = {};
            std::vector<dealii::CellId> cells;
            for (typename dealii::Triangulation<dim>::cell_iterator cell :
                 triangulation.active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  while (cell->level() > level_to_store_particles)
                    cell = cell->parent();
                  cells.push_back(cell->id());
                }
            auto new_end = std::ranges::unique(cells);
            cells.erase(new_end.begin(), cells.end());
            reference_cell_to_rank_send[rank] = cells;
          }
      }

    EXPECT_EQ(reference_cell_to_rank_send, partitioner_cell_to_rank_send);
  }
}