#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/cell_id.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

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

TEST_F(ParticleDataStructureTest, GhostParticles)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  std::vector<dealii::Point<dim, number>> obstacle_locations = {
    dealii::Point<dim, number>(0.3, 0.3),
    dealii::Point<dim, number>(0.8, 0.2),
    dealii::Point<dim, number>(0.6, 0.4),
    dealii::Point<dim, number>(0.8, 0.7)};

  std::vector<std::vector<number>> obstacle_properties;
  obstacle_properties.resize(obstacle_locations.size());
  number mass = 1;
  for (std::vector<number> &properties : obstacle_properties)
    {
      properties.resize(MeltPoolDG::SphericalParticle<dim, number>::n_obstacle_properties);
      properties[MeltPoolDG::SphericalParticle<dim, number>::Properties::radius] = 0.4;
      properties[MeltPoolDG::SphericalParticle<dim, number>::Properties::mass]   = mass;
      mass++;
    }

  obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
  obstacle_data_structure.sort_particles_into_subdomains_and_cells();

  std::map<typename dealii::Triangulation<dim>::cell_iterator,
           std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
    ghost_particles = obstacle_data_structure.get_cell_to_ghost_particle_cache();

  std::cout << "Rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            << ": Locally relevant particles: "
            << obstacle_data_structure.n_locally_relevant_particles() << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  unsigned int n_ghost_particles = 0;
  for (const auto &[cell, particles] : ghost_particles)
    n_ghost_particles += particles.size();

  unsigned n_expected_ghost_particles = obstacle_locations.size();

  // Ranks 1, 2, 3 and 6 own a particle and therefore have 3 ghost particles, while ranks 0, 4, 5
  // and 7 do not own any particle and therefore have 4 ghost particles.
  if (MeltPoolDG::Utils::contains<std::vector<unsigned int>>(
        {{1, 2, 3, 6}}, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)))
    {
      n_expected_ghost_particles -= 1;
    }

  {
    SCOPED_TRACE("Check ghost particles on rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));
    EXPECT_EQ(n_expected_ghost_particles, n_ghost_particles);
  }
}

TEST_F(ParticleDataStructureTest, ParticleNumberInfo)
{
  std::vector<dealii::Point<dim, number>> obstacle_locations = {
    dealii::Point<dim, number>(0.3, 0.3),
    dealii::Point<dim, number>(0.8, 0.2),
    dealii::Point<dim, number>(0.6, 0.4),
    dealii::Point<dim, number>(0.8, 0.7)};

  std::vector<std::vector<number>> obstacle_properties;
  obstacle_properties.resize(obstacle_locations.size());
  number mass = 1;
  for (std::vector<number> &properties : obstacle_properties)
    {
      properties.resize(MeltPoolDG::SphericalParticle<dim, number>::n_obstacle_properties);
      properties[MeltPoolDG::SphericalParticle<dim, number>::Properties::radius] = 0.4;
      properties[MeltPoolDG::SphericalParticle<dim, number>::Properties::mass]   = mass;
      mass++;
    }

  obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
  obstacle_data_structure.sort_particles_into_subdomains_and_cells();

  std::map<typename dealii::Triangulation<dim>::cell_iterator,
           std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
    ghost_particles = obstacle_data_structure.get_cell_to_ghost_particle_cache();

  {
    SCOPED_TRACE("Check particle numbers on rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));
    EXPECT_EQ(obstacle_data_structure.n_locally_relevant_particles(), obstacle_locations.size());
    EXPECT_EQ(obstacle_data_structure.n_ghost_particles(),
              obstacle_locations.size() - obstacle_data_structure.n_locally_owned_particles());
  }
}

TEST_F(ParticleDataStructureTest, PartitionerReceiverRanks)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  constexpr int  level_to_store_particles = 1;

  MeltPoolDG::LevelCellPartitioner<dim> partitioner(triangulation);
  partitioner.reinit(level_to_store_particles);

  {
    SCOPED_TRACE("Check for correct receiver of data from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));

    // These are the owner ranks of the cells on level 1.
    std::vector<int> reference_receiver_ranks(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    std::iota(reference_receiver_ranks.begin(), reference_receiver_ranks.end(), 0);

    auto range = std::ranges::remove(reference_receiver_ranks,
                                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    reference_receiver_ranks.erase(range.begin(), range.end());

    std::vector<int> partitioner_receiver_ranks = partitioner.get_particle_receiver_ranks();
    std::ranges::sort(partitioner_receiver_ranks);

    EXPECT_EQ(reference_receiver_ranks, partitioner_receiver_ranks);
  }
}

TEST_F(ParticleDataStructureTest, PartitionerSenderRanks)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  constexpr int level_to_store_particles = 1;

  MeltPoolDG::LevelCellPartitioner<dim> partitioner(triangulation);
  partitioner.reinit(level_to_store_particles);

  {
    SCOPED_TRACE("Check for correct sender of data from rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)));

    std::vector<int> reference_sender_ranks;
    reference_sender_ranks.resize(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    std::iota(reference_sender_ranks.begin(), reference_sender_ranks.end(), 0);
    auto range = std::ranges::remove(reference_sender_ranks,
                                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    reference_sender_ranks.erase(range.begin(), range.end());

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

/*
TEST(ParticleDataStructureTest, PartitionerSenderAndReceiverRanks)
{
  // The test is designed to run with 8 MPI processes.
  ASSERT_EQ(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 8);

  constexpr int level_to_store_particles = 1;


}
  */

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
