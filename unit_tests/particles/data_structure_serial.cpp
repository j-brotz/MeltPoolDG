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

using number = double;

static constexpr int                                                    dim = 2;
static constexpr MeltPoolDG::SphericalParticle<dim, number>::Properties mass_property =
  MeltPoolDG::SphericalParticle<dim, number>::Properties::mass;


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
    triangulation.refine_global(3);
  }
  dealii::parallel::distributed::Triangulation<dim> triangulation;

  dealii::MappingQ<dim> mapping;

  dealii::TimerOutput timer;

  MeltPoolDG::CellListParticleHandler<dim, number, MeltPoolDG::SphericalParticle<dim, number>>
    obstacle_data_structure;
};

TEST_F(ParticleDataStructureTest, ContactParticles)
{
  constexpr number particle_radius = 0.2;

  std::vector<dealii::Point<dim, number>> obstacle_locations = {
    dealii::Point<dim, number>(0.4, 0.5), dealii::Point<dim, number>(0.79, 0.5)};

  std::vector<std::vector<number>> obstacle_properties;
  obstacle_properties.resize(obstacle_locations.size());
  number mass = 1;
  for (std::vector<number> &properties : obstacle_properties)
    {
      properties.resize(MeltPoolDG::SphericalParticle<dim, number>::n_obstacle_properties);
      properties[MeltPoolDG::SphericalParticle<dim, number>::Properties::radius] = particle_radius;
      properties[mass_property]                                                  = mass;
      mass += 1;
    }

  obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
  obstacle_data_structure.reinit(particle_radius);
  obstacle_data_structure.sort_particles_into_subdomains_and_cells();

  for (const auto &particle : obstacle_data_structure.locally_owned_particle_range())
    {
      boost::container::small_vector<MeltPoolDG::DEMParticleAccessor<dim, number>, 3 *dim>
        contact_particles = obstacle_data_structure.contact_particles(particle, 0.0);
      ASSERT_EQ(contact_particles.size(), 1);
      EXPECT_DOUBLE_EQ(contact_particles[0].get_property(mass_property),
                       particle.get_property(mass_property) == 1 ? 2 : 1);
    }
}
