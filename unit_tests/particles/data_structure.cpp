#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <iostream>

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
