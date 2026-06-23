#include <gtest/gtest.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/triangulation_utils.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

constexpr int dim = 2;
using number      = double;

class LevelCommunicationPatternTest : public ::testing::Test
{
protected:
  LevelCommunicationPatternTest()
    : triangulation(
        MPI_COMM_WORLD,
        dealii::Triangulation<dim>::MeshSmoothing::none,
        dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy)
    , partitioner(triangulation)
  {
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(4);
  }

  dealii::parallel::distributed::Triangulation<dim> triangulation;
  MeltPoolDG::LevelCommunicationPattern<dim>        partitioner;
};

/**
 * Check that rebuilding the communication pattern with the same level does not change the pattern.
 */
TEST_F(LevelCommunicationPatternTest, RebuildIsIdempotent)
{
  constexpr int level_to_store_particles = 1;
  partitioner.build_pattern(level_to_store_particles);


  const auto send_ranks_first    = partitioner.send_ranks();
  const auto receive_ranks_first = partitioner.receive_ranks();
  const auto cells_send_first    = partitioner.cells_to_send();
  const auto cells_recv_first    = partitioner.cells_to_receive();

  partitioner.build_pattern(level_to_store_particles);

  EXPECT_EQ(partitioner.send_ranks(), send_ranks_first);
  EXPECT_EQ(partitioner.receive_ranks(), receive_ranks_first);
  EXPECT_EQ(partitioner.cells_to_send(), cells_send_first);
  EXPECT_EQ(partitioner.cells_to_receive(), cells_recv_first);
}

/**
 * Check that the current rank is not listed as a send target or receive source in the communication
 * pattern. This would imply to send and receive data to and from itself, which should not happen.
 * Note that this test does not check that the communication pattern is correct, but only that it
 * does not contain self-communication.
 */
TEST_F(LevelCommunicationPatternTest, NoSelfCommunication)
{
  constexpr int level_to_store_particles = 1;
  partitioner.build_pattern(level_to_store_particles);

  const unsigned int current_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const auto send_ranks    = partitioner.send_ranks();
  const auto receive_ranks = partitioner.receive_ranks();

  EXPECT_EQ(std::ranges::find(send_ranks, current_rank), send_ranks.end())
    << "Rank " << current_rank << " lists itself as a send target.";
  EXPECT_EQ(std::ranges::find(receive_ranks, current_rank), receive_ranks.end())
    << "Rank " << current_rank << " lists itself as a receive source.";
}

/**
 * Check that the communication pattern is consistent between senders and receivers. This means that
 * if a rank A lists rank B as a send target, then rank B should list rank A as a receive source,
 * and vice versa. This test does not check that the communication pattern is correct, but only that
 * it is consistent between senders and receivers.
 */
TEST_F(LevelCommunicationPatternTest, SendReceiveMatch)
{
  constexpr int level_to_store_particles = 1;
  partitioner.build_pattern(level_to_store_particles);

  const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Gather every rank's send list onto all rank
  std::vector<unsigned int>              my_send_ranks = partitioner.send_ranks();
  std::vector<std::vector<unsigned int>> all_send_ranks =
    dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_send_ranks);

  // Same check from the receive side.
  for (unsigned int source : partitioner.receive_ranks())
    {
      EXPECT_NE(std::ranges::find(all_send_ranks[source], my_rank), all_send_ranks[source].end())
        << "Rank " << my_rank << " expects to receive from rank " << source << " but rank "
        << source << " does not list us as a send target.";
    }
}

/**
 * Check that the communication pattern is symmetric. This means that if a rank A lists rank B as a
 * send target, then rank B should also list rank A as a send target. This must be the case as we
 * only send and receive data between ranks if tboth share a level cell for which they have at least
 * one locally owned active descendant, and this is a symmetric relation. This test does not check
 * that the communication pattern is correct, but only that it is symmetric between senders.
 */
TEST_F(LevelCommunicationPatternTest, SendSymmetry)
{
  constexpr int level_to_store_particles = 1;
  partitioner.build_pattern(level_to_store_particles);


  const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Gather every rank's send list onto all ranks.
  std::vector<unsigned int>              my_send_ranks = partitioner.send_ranks();
  std::vector<std::vector<unsigned int>> all_send_ranks =
    dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_send_ranks);

  // For each rank R that we send to, verify R also sends to us.
  for (unsigned int target : my_send_ranks)
    {
      EXPECT_NE(std::ranges::find(all_send_ranks[target], my_rank), all_send_ranks[target].end())
        << "Rank " << my_rank << " sends to rank " << target << " but rank " << target
        << " does not send back.";
    }
}

/**
 * Same test as SendSymmetry but from the receive side. This means that if a rank A lists rank B as
 * a receive source, then rank B should also list rank A as a receive source.
 */
TEST_F(LevelCommunicationPatternTest, ReceiveSymmetry)
{
  constexpr int level_to_store_particles = 1;
  partitioner.build_pattern(level_to_store_particles);


  const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Gather every rank's receiver list onto all ranks.
  std::vector<unsigned int>              my_receive_ranks = partitioner.receive_ranks();
  std::vector<std::vector<unsigned int>> all_receive_ranks =
    dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_receive_ranks);

  // For each rank R that we receive from, verify R also receives from us.
  for (unsigned int target : my_receive_ranks)
    {
      EXPECT_NE(std::ranges::find(all_receive_ranks[target], my_rank),
                all_receive_ranks[target].end())
        << "Rank " << my_rank << " receives from rank " << target << " but rank " << target
        << " does not receive from us.";
    }
}

/**
 * This test verifies the receiver ranks in the communication pattern. A 2D cube is globally refined
 * four times, resulting in 256 active cells on the finest level. The level of interest is level 1,
 * which contains four cells. Consequently, there are four distinct ranks owning the level-1 cells.
 * Each of these owners must send data to other ranks, since every rank owns active descendant cells
 * of either the level-1 cell itself or one of its neighboring level-1 cells. As a result, each rank
 * is expected to receive data from all level-1 cell owners except itself. For this scenario, this
 * test verifies that the receiver ranks in the communication pattern match this expected
 * communication topology.
 *
 * @note
 */
TEST_F(LevelCommunicationPatternTest, ReceiverRanksTopology)
{
  // We check that the number of MPI processes is less than the number of global active cells
  // divided by a safety factor of 4, which ensures that there are enough cells for each rank.
  ASSERT_LE(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
            triangulation.n_global_active_cells() / 4)
    << "This test is designed to run with less than " << triangulation.n_global_active_cells() / 4
    << " MPI processes.";

  constexpr int level_to_store_particles = 1;
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

/**
 * Using the scenario described in the test above, this test verifies the sender ranks in the
 * communication pattern. Each rank owning a level-1 cell is expected to send data to all other
 * ranks, since every rank owns active descendant cells of either that level-1 cell or one of its
 * neighboring level-1 cells. This test verifies that the sender ranks in the communication pattern
 * match this expected communication topology. In addition, it checks that the set of cells
 * scheduled to be sent to each receiving rank is correct.
 */
TEST_F(LevelCommunicationPatternTest, SenderRanksTopology)
{
  // We check that the number of MPI processes is less than the number of global active cells
  // divided by a safety factor of 4, which ensures that there are enough cells for each rank.
  ASSERT_LE(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
            triangulation.n_global_active_cells() / 4)
    << "This test is designed to run with less than " << triangulation.n_global_active_cells() / 4
    << " MPI processes.";

  constexpr int level_to_store_particles = 1;
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
