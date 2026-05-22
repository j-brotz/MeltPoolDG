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

TEST_F(ParticleDataStructureTest, CompressParticleData)
{
  std::vector<dealii::Point<dim, number>> obstacle_locations = {
    dealii::Point<dim, number>(0.3, 0.3)};

  std::vector<std::vector<number>> obstacle_properties;
  obstacle_properties.resize(obstacle_locations.size());
  for (std::vector<number> &properties : obstacle_properties)
    {
      properties.resize(MeltPoolDG::SphericalParticle<dim, number>::n_obstacle_properties);
      properties[MeltPoolDG::SphericalParticle<dim, number>::Properties::radius] = 0.2;
    }

  obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);

  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  for (auto &particle : obstacle_data_structure.locally_owned_particle_range())
    {
      for (int d = 0; d < dim; ++d)
        particle.force(d) = (d + 1) * (rank);
      for (int d = 0; d < MeltPoolDG::axial_dim<dim>; ++d)
        particle.torque(d) = (d + 1) * (rank + 1);
    }

  for (auto &particle : obstacle_data_structure.ghost_particle_range())
    {
      for (int d = 0; d < dim; ++d)
        particle.force(d) = (d + 1) * (rank);
      for (int d = 0; d < MeltPoolDG::axial_dim<dim>; ++d)
        particle.torque(d) = (d + 1) * (rank + 1);
    }

  obstacle_data_structure.compress();

  // Compute expected values
  dealii::Tensor<1, dim>                        expected_force;
  dealii::Tensor<1, MeltPoolDG::axial_dim<dim>> expected_torque;
  for (unsigned int r = 0; r < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++r)
    {
      for (int d = 0; d < dim; ++d)
        expected_force[d] += (d + 1) * r;
      for (int d = 0; d < MeltPoolDG::axial_dim<dim>; ++d)
        expected_torque[d] += (d + 1) * (r + 1);
    }

  // Check for validity of test. It must only exist one global particle in the domain.
  ASSERT_EQ(obstacle_data_structure.n_global_particles(), 1);

  // Compare the values
  for (auto &particle : obstacle_data_structure.locally_owned_particle_range())
    {
      {
        SCOPED_TRACE("Compressed particle force");
        for (int d = 0; d < dim; ++d)
          EXPECT_DOUBLE_EQ(particle.force(d), expected_force[d]);
      }
      {
        SCOPED_TRACE("Compressed particle torque");
        for (int d = 0; d < MeltPoolDG::axial_dim<dim>; ++d)
          EXPECT_DOUBLE_EQ(particle.torque(d), expected_torque[d]);
      }
    }
}

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

  std::unordered_map<dealii::types::global_cell_index,
                     std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
    ghost_particles = obstacle_data_structure.get_cell_to_ghost_particle_cache();

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

  std::unordered_map<dealii::types::global_cell_index,
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
