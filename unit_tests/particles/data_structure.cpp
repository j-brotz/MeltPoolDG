#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/cell_id.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <algorithm>
#include <iostream>

#include "mpi.h"

static constexpr int dim = 2;
using number             = double;

class ParticleDataStructureTest : public ::testing::Test
{
protected:
  ParticleDataStructureTest()
    : triangulation(
        MPI_COMM_WORLD,
        dealii::Triangulation<dim>::MeshSmoothing::none,
        dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy)
    , mapping(1)
    , timer(std::cout, dealii::TimerOutput::never, dealii::TimerOutput::wall_times)
    , obstacle_data_structure(triangulation, mapping, timer)
  {
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(4);
  }
  dealii::parallel::distributed::Triangulation<dim> triangulation;

  dealii::MappingQ<dim> mapping;

  dealii::TimerOutput timer;

  MeltPoolDG::ObstacleCompleteDomainSearch<dim, number, MeltPoolDG::SphericalParticle<dim, number>>
    obstacle_data_structure;
};

/*
TEST_F(ParticleDataStructureTest, Reinit)
{
  constexpr int level_to_store_particles = 1;

  std::vector<dealii::Point<dim, number>> obstacle_locations = {
    dealii::Point<dim, number>(0.3, 0.3),
    dealii::Point<dim, number>(0.8, 0.2),
    dealii::Point<dim, number>(0.6, 0.4),
    dealii::Point<dim, number>(0.8, 0.7)};

  std::vector<std::vector<number>> obstacle_properties;
  obstacle_properties.reserve(obstacle_locations.size());
  for (std::vector<number> &properties : obstacle_properties)
    {
      properties.resize(MeltPoolDG::SphericalParticle<dim, number>::n_obstacle_properties);
      properties[MeltPoolDG::SphericalParticle<dim, number>::Properties::radius] = 0.6;
    }

  std::cout << "I am here 0" << std::endl;

  obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
  std::cout << "I am here 1" << std::endl;
  obstacle_data_structure.sort_particles_into_subdomains_and_cells();

  std::cout << "I am here 2" << std::endl;
  for (typename dealii::Triangulation<dim>::cell_iterator cell :
       triangulation.cell_iterators_on_level(level_to_store_particles))
    if (cell->is_locally_owned_on_level())
      {
        EXPECT_EQ(obstacle_locations.size(),
                  obstacle_data_structure.get_obstacles_in_cell(*cell).size());
      }
}
      */

TEST_F(ParticleDataStructureTest, PartitionerReceiverRanks)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  constexpr int  level_to_store_particles = 1;
  dealii::CellId coarse_cell_id           = triangulation.begin(0)->id();

  MeltPoolDG::LevelCellPartitioner<dim> partitioner(triangulation, level_to_store_particles);
  partitioner.reinit();

  {
    SCOPED_TRACE("Check for correct receiver of data from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));

    // These are the owner ranks of the cells on level 1.
    std::vector<int> reference_receiver_ranks = {0, 2, 4, 6};

    // If I am one of the owner I do not need to receive data from myself, so I remove myself from
    // the list of reference ranks.
    if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) % 2 == 0)
      {
        auto range = std::ranges::remove(reference_receiver_ranks,
                                         dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
        reference_receiver_ranks.erase(range.begin(), range.end());
      }

    std::vector<int> partitioner_receiver_ranks = partitioner.get_particle_receiver_ranks();
    std::ranges::sort(partitioner_receiver_ranks);

    EXPECT_EQ(reference_receiver_ranks, partitioner_receiver_ranks);
  }

  {
    SCOPED_TRACE("Check that correct cells are received from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));
    std::map<int, std::vector<dealii::CellId>> partitioner_cell_to_rank_receive =
      partitioner.get_cell_to_rank_receive();


    std::map<int, std::vector<dealii::CellId>> reference_cell_to_rank_receive;

    int i = 0;
    for (unsigned rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++rank)
      {
        if (rank % 2 == 0 and rank != dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
          {
            std::uint8_t child_indices           = i;
            reference_cell_to_rank_receive[rank] = {
              dealii::CellId(coarse_cell_id.get_coarse_cell_id(), 1u, &child_indices)};
          }
        if (rank % 2 == 0)
          ++i;
      }

    EXPECT_EQ(reference_cell_to_rank_receive, partitioner_cell_to_rank_receive);
  }
}

TEST_F(ParticleDataStructureTest, PartitionerSenderRanks)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  constexpr int level_to_store_particles = 1;

  MeltPoolDG::LevelCellPartitioner<dim> partitioner(triangulation, level_to_store_particles);
  partitioner.reinit();

  {
    SCOPED_TRACE("Check for correct sender of data from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));

    // If I am no owner of any cell on the level I do not need to send data to any other rank, so
    // the list of reference ranks is empty.
    std::vector<int> reference_sender_ranks;

    // If I am one of the owner I need to send my information to all other ranks
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) % 2 == 0)
      {
        reference_sender_ranks.resize(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
        std::iota(reference_sender_ranks.begin(), reference_sender_ranks.end(), 0);
        auto range = std::ranges::remove(reference_sender_ranks,
                                         dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
        reference_sender_ranks.erase(range.begin(), range.end());
      }

    std::vector<int> partitioner_sender_ranks = partitioner.get_particle_sender_ranks();
    std::ranges::sort(partitioner_sender_ranks);

    EXPECT_EQ(reference_sender_ranks, partitioner_sender_ranks);
  }

  {
    SCOPED_TRACE("Check that correct cells are sent from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));
    std::map<int, std::vector<dealii::CellId>> partitioner_cell_to_rank_send =
      partitioner.get_cell_to_rank_send();


    std::map<int, std::vector<dealii::CellId>> reference_cell_to_rank_send;
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) % 2 == 0)
      {
        for (unsigned rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             ++rank)
          {
            if (rank != dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
              {
                reference_cell_to_rank_send[rank] = {};
                for (typename dealii::Triangulation<dim>::cell_iterator cell :
                     triangulation.cell_iterators_on_level(level_to_store_particles))
                  if (cell->is_locally_owned_on_level())
                    reference_cell_to_rank_send[rank].push_back(cell->id());
              }
          }
      }

    EXPECT_EQ(reference_cell_to_rank_send, partitioner_cell_to_rank_send);
  }
}

/*
TEST_F(ParticleDataStructureTest, ArtificalCellsOnLevelZero)
{
  if (triangulation.begin(0)->is_artificial_on_level())
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " only has ARTIFICIAL cells on level 0." << std::endl;
    }
  else if (triangulation.begin(0)->is_ghost_on_level())
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " only has GHOST cells on level 0." << std::endl;
    }
  else if (triangulation.begin(0)->is_locally_owned_on_level())
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " only has LOCALLY OWNED cells on level 0." << std::endl;
    }
  else
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " has no cells on level 0." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(ParticleDataStructureTest, CellID)
{
  std::array<std::uint8_t, 3>                        child_indices = {{0, 0, 0}};
  dealii::CellId                                     cell_id(0, 3, child_indices.data());
  typename dealii::Triangulation<dim>::cell_iterator cell;
  if (triangulation.contains_cell(cell_id))
    cell = triangulation.create_cell_iterator(cell_id);
  else
    {
      auto child_indices = cell_id.get_child_indices();
      auto new_cell_id   = dealii::CellId(cell_id.get_coarse_cell_id(),
                                        child_indices.size() - 1,
                                        child_indices.data());
      cell               = triangulation.create_cell_iterator(new_cell_id);
    }
  if (cell->is_artificial_on_level())
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " only has ARTIFICIAL cells on level 0." << std::endl;
    }
  else if (cell->is_ghost_on_level())
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " only has GHOST cells on level 0." << std::endl;
    }
  else if (cell->is_locally_owned_on_level())
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " only has LOCALLY OWNED cells on level 0." << std::endl;
    }
  else
    {
      std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " has no cells on level 0." << std::endl;
    }

  MPI_Barrier(MPI_COMM_WORLD);
}
  */
