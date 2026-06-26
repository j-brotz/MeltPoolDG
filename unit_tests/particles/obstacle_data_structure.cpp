#include <gtest/gtest.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>

#include <limits>
#include <string>
#include <vector>

#include "../test_utils/utils.hpp"

namespace
{
  constexpr int dim = 3;
  using namespace MeltPoolDG;
} // namespace

/**
 * A struct to hold the properties of a reference particle for comparison purposes.
 */
struct ReferenceParticleProperties
{
  bool
  operator==(const ReferenceParticleProperties &other) const
  {
    return (location == other.location) and (radius == other.radius) and
           (density == other.density) and (velocity == other.velocity) and
           (force == other.force) and (torque == other.torque);
  }

  dealii::Point<dim, double>                location;
  double                                    radius;
  double                                    density;
  dealii::Tensor<1, dim, double>            velocity;
  dealii::Tensor<1, dim, double>            force;
  dealii::Tensor<1, axial_dim<dim>, double> torque;
  bool                                      is_ghost_on_other_mpi_process = false;
};

/**
 * Helper function, which sorts a vector of ReferenceParticleProperties based on the radius of the
 * particles. This is used to ensure that the particles are sorted in a consistent order for
 * comparison purposes.
 *
 * @note To guarantee consistent ordering, all particles must have unique radii. If two particles
 * have the same radius, the sorting order is not guaranteed.
 */
void
sort_reference_particles(std::vector<ReferenceParticleProperties> &particles)
{
  std::ranges::sort(particles,
                    [](const ReferenceParticleProperties &a, const ReferenceParticleProperties &b) {
                      return a.radius < b.radius;
                    });
}

/**
 * This function takes a range of particles, i.e., a range of DEMParticleAccessor objects, and
 * creates a vector of ReferenceParticleProperties.
 *
 * @param particle_range The range of DEMParticleAccessor objects representing the particles.
 */
template <typename ParticleRange>
std::vector<ReferenceParticleProperties>
make_reference_particles_from_particle_range(const ParticleRange &particle_range)
{
  std::vector<ReferenceParticleProperties> reference_particles;
  for (const DEMParticleAccessor<dim, double> &particle : particle_range)
    {
      ReferenceParticleProperties particle_properties;
      particle_properties.location = particle.get_location();
      particle_properties.radius   = particle.radius();
      particle_properties.density  = particle.density();
      particle_properties.force    = particle.force();
      particle_properties.torque   = particle.torque();
      for (unsigned int d = 0; d < dim; ++d)
        particle_properties.velocity[d] = particle.linear_velocity(d);
      reference_particles.push_back(particle_properties);
    }
  return reference_particles;
}

/**
 * This function compares a range of particles with a vector of reference particles. It checks that
 * the number of particles in the range matches the number of reference particles and that each
 * particle's properties match the corresponding reference particle's properties. The comparison is
 * done after sorting both the particle range and the reference particles based on their radius to
 * ensure consistent ordering.
 *
 * @param particle_range The range of DEMParticleAccessor objects representing the particles to be
 * compared.
 * @param reference_particles The vector of reference particles to compare against.
 */
template <typename ParticleRange>
void
compare_particle_range_with_reference_particles(
  const ParticleRange                     &particle_range,
  std::vector<ReferenceParticleProperties> reference_particles)
{
  std::vector<ReferenceParticleProperties> particle_properties_from_range =
    make_reference_particles_from_particle_range(particle_range);

  sort_reference_particles(reference_particles);
  sort_reference_particles(particle_properties_from_range);

  ASSERT_EQ(reference_particles.size(), particle_properties_from_range.size())
    << "Particle count mismatch: expected " << reference_particles.size() << ", got "
    << particle_properties_from_range.size();
  EXPECT_EQ(reference_particles, particle_properties_from_range);
}

/**
 * This test fixture sets up a distributed triangulation and an obstacle data structure for testing
 * the functionality of the obstacle data structure in a parallel environment. It provides utility
 * functions to insert particles into the data structure and to create reference particle properties
 * for comparison in the tests.
 *
 * The test scenario consist of a triangulation being  a 2x1x1 box [0,2]x[0,1]x[0,1], split into two
 * coarse cells along x (coarse cell 0: x in [0,1], coarse cell 1: x in [1,2]), then globally
 * refined 4 times. Each coarse cell therefore becomes a 16x16x16 grid of fine cells. In a 2-rank
 * run, rank 0 owns coarse cell 0 and rank 1 owns coarse cell 1. The following sketch illustrates
 * the setup of the triangulation:
 *
 *   y=1 +-----------------------+-----------------------+
 *       |                       |                       |
 *       |     coarse cell 0     |     coarse cell 1     |
 *       |        (rank 0)       |        (rank 1)       |
 *       |                       |                       |
 *   y=0 +-----------------------+-----------------------+
 *      x=0                     x=1                     x=2
 *
 * In addition a set of particles is defined for each coarse cell, which consist of both particles
 * that are only relevant for the locally owned domain, and particles that are relevant for both,
 * the locally owned domain and the ghost domain on the other rank.
 */
class ObstacleDataStructureTest : public ::testing::Test
{
protected:
  ObstacleDataStructureTest()
    : triangulation(
        MPI_COMM_WORLD,
        dealii::Triangulation<dim>::MeshSmoothing::none,
        dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy)
    , mapping(1)
    , obstacle_data_structure(triangulation, mapping)
  {
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      std::vector<unsigned int>{2, 1, 1},
                                                      dealii::Point<dim>(0., 0., 0.),
                                                      dealii::Point<dim>(2., 1., 1.));
    triangulation.refine_global(4);
  }

  /**
   * Given a vector of ReferenceParticleProperties, this function inserts the particles into the
   * obstacle data structure. It extracts the locations and properties of the particles and calls
   * the `insert_global_particles()` function of the obstacle data structure to perform the
   * insertion.
   *
   * @param particles The vector of ReferenceParticleProperties representing the particles to be
   * inserted.
   */
  void
  insert_particles(const std::vector<ReferenceParticleProperties> &particles)
  {
    std::vector<dealii::Point<dim, double>> obstacle_locations;
    std::vector<std::vector<double>>        obstacle_properties;

    for (const auto &particle : particles)
      {
        obstacle_locations.push_back(particle.location);
        obstacle_properties.push_back(make_particle_properties_vector(particle));
      }

    obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
  }

  /**
   * Given a ReferenceParticleProperties object, this function creates a vector of properties for
   * the particle.
   *
   * @param particle The ReferenceParticleProperties object representing the particle.
   * @return A vector of doubles representing the properties of the particle.
   */
  std::vector<double>
  make_particle_properties_vector(const ReferenceParticleProperties &particle) const
  {
    std::vector<double> properties(SphericalParticle<dim, double>::n_obstacle_properties, 0.0);
    properties[SphericalParticle<dim, double>::Properties::radius]  = particle.radius;
    properties[SphericalParticle<dim, double>::Properties::density] = particle.density;
    for (unsigned int d = 0; d < dim; ++d)
      {
        properties[SphericalParticle<dim, double>::Properties::velocity + d] = particle.velocity[d];
        properties[SphericalParticle<dim, double>::Properties::force + d]    = particle.force[d];
      }

    for (unsigned int d = 0; d < axial_dim<dim>; ++d)
      properties[SphericalParticle<dim, double>::Properties::torque + d] = particle.torque[d];

    if constexpr (dim == 3)
      {
        properties[SphericalParticle<dim, double>::Properties::volume] =
          4.0 / 3.0 * M_PI *
          std::pow(properties[SphericalParticle<dim, double>::Properties::radius], 3);
        properties[SphericalParticle<dim, double>::Properties::mass] =
          properties[SphericalParticle<dim, double>::Properties::volume] *
          properties[SphericalParticle<dim, double>::Properties::density];
        properties[SphericalParticle<dim, double>::Properties::moment_of_inertia] =
          0.4 * properties[SphericalParticle<dim, double>::Properties::mass] *
          std::pow(properties[SphericalParticle<dim, double>::Properties::radius], 2);
      }
    else if constexpr (dim == 2)
      {
        properties[SphericalParticle<dim, double>::Properties::volume] =
          M_PI * std::pow(properties[SphericalParticle<dim, double>::Properties::radius], 2);
        properties[SphericalParticle<dim, double>::Properties::mass] =
          properties[SphericalParticle<dim, double>::Properties::volume] *
          properties[SphericalParticle<dim, double>::Properties::density];
        properties[SphericalParticle<dim, double>::Properties::moment_of_inertia] =
          0.5 * properties[SphericalParticle<dim, double>::Properties::mass] *
          std::pow(properties[SphericalParticle<dim, double>::Properties::radius], 2);
      }
    return properties;
  }

  /// The triangulation used for storing the particles on.
  dealii::parallel::distributed::Triangulation<dim> triangulation;

  /// Mapping required for the obstacle data structure.
  dealii::MappingQ<dim> mapping;

  /// The obstacle data structure used for storing and managing particles and to be tested by the
  /// tests building up on this test fixture.
  CellListParticleHandler<dim, double, MeltPoolDG::SphericalParticle<dim, double>>
    obstacle_data_structure;

  /// Reference particles for the first coarse cell (cell 0) used in the tests.
  std::vector<ReferenceParticleProperties> reference_particles_coarse_cell_0 = {
    {.location                      = dealii::Point<dim>(0.99, 0.49, 0.45),
     .radius                        = 0.15,
     .density                       = static_cast<double>('a') + 380,
     .velocity                      = dealii::Tensor<1, dim, double>({2.0, 3.0, 4.0}),
     .force                         = dealii::Tensor<1, dim, double>({1.0, 0.0, 2.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({3.0, 0.0, 7.0}),
     .is_ghost_on_other_mpi_process = true},
    {.location                      = dealii::Point<dim>(0.7, 0.65, 0.6),
     .radius                        = 0.195,
     .density                       = 320,
     .velocity                      = dealii::Tensor<1, dim, double>({4.0, 1.0, 5.0}),
     .force                         = dealii::Tensor<1, dim, double>({0.0, 1.0, 0.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({0.0, 3.0, 4.0}),
     .is_ghost_on_other_mpi_process = true},
    {.location                      = dealii::Point<dim>(0.25, 0.6, 0.45),
     .radius                        = 0.14,
     .density                       = 330,
     .velocity                      = dealii::Tensor<1, dim, double>({0.0, 3.0, 1.0}),
     .force                         = dealii::Tensor<1, dim, double>({0.7, 2.0, 0.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({3.0, 3.0, 4.0}),
     .is_ghost_on_other_mpi_process = false},
    {.location                      = dealii::Point<dim>(0.1, 0.5, 0.5),
     .radius                        = 0.17,
     .density                       = 340,
     .velocity                      = dealii::Tensor<1, dim, double>({0.0, 7.0, 2.0}),
     .force                         = dealii::Tensor<1, dim, double>({3.0, 7.0, 0.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({7.0, 3.0, 7.0}),
     .is_ghost_on_other_mpi_process = false}};

  /// Reference particles for the second coarse cell (cell 1) used in the tests.
  std::vector<ReferenceParticleProperties> reference_particles_coarse_cell_1 = {
    {.location                      = dealii::Point<dim>(1.1, 0.5, 0.45),
     .radius                        = 0.11,
     .density                       = static_cast<double>('b') + 747,
     .velocity                      = dealii::Tensor<1, dim, double>({1.0, 8.0, 0.0}),
     .force                         = dealii::Tensor<1, dim, double>({7.0, 8.0, 7.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({3.0, 4.0, 0.0}),
     .is_ghost_on_other_mpi_process = true},
    {.location                      = dealii::Point<dim>(1.2, 0.45, 0.52),
     .radius                        = 0.2,
     .density                       = 737,
     .velocity                      = dealii::Tensor<1, dim, double>({3.0, 1.0, 0.0}),
     .force                         = dealii::Tensor<1, dim, double>({7.0, 0.0, 7.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({3.0, 2.0, 0.0}),
     .is_ghost_on_other_mpi_process = true},
    {.location                      = dealii::Point<dim>(1.4, 0.8, 0.1),
     .radius                        = 0.15,
     .density                       = 777,
     .velocity                      = dealii::Tensor<1, dim, double>({5.0, 0.0, 1.0}),
     .force                         = dealii::Tensor<1, dim, double>({3.0, 5.0, 0.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({9.0, 0.0, 0.0}),
     .is_ghost_on_other_mpi_process = true},
    {.location                      = dealii::Point<dim>(1.8, 0.5, 0.5),
     .radius                        = 0.18,
     .density                       = 787,
     .velocity                      = dealii::Tensor<1, dim, double>({0.0, 7.0, 1.0}),
     .force                         = dealii::Tensor<1, dim, double>({0.0, 7.0, 0.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({7.0, 0.0, 7.0}),
     .is_ghost_on_other_mpi_process = false},
    {.location                      = dealii::Point<dim>(1.6, 0.5, 0.5),
     .radius                        = 0.1202,
     .density                       = 707,
     .velocity                      = dealii::Tensor<1, dim, double>({5.0, 0.0, 1.0}),
     .force                         = dealii::Tensor<1, dim, double>({2.0, 0.0, 0.0}),
     .torque                        = dealii::Tensor<1, axial_dim<dim>, double>({0.0, 3.0, 0.0}),
     .is_ghost_on_other_mpi_process = false}};

  /// A convenient way to access the reference particles for each coarse cell based on the rank of
  /// the MPI process. The index of the outer vector corresponds to the rank of the MPI process, and
  /// the inner vector contains the reference particles for that rank's locally owned coarse cell.
  /// This allows for easy access to the reference particles based on the rank of the MPI process
  /// during testing. It should only be used when two two MPI processes are used for the test, as
  /// the test scenario is designed for a 2-rank run.
  std::vector<std::vector<ReferenceParticleProperties>> reference_particles_on_rank =
    {reference_particles_coarse_cell_0, reference_particles_coarse_cell_1};
};

/**
 * Verifies that the `insert_global_particles()` function correctly inserts particles into the data
 * structure across multiple MPI processes. In this test, we insert particles from two different MPI
 * processes and check that the locally owned particle range on each process matches the expected
 * reference particles including the specified particle properties. The test ensures that the
 * particles are correctly distributed and accessible on their respective owning ranks.
 */
TEST_F(ObstacleDataStructureTest, GlobalInsertParticles)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);

  compare_particle_range_with_reference_particles(
    obstacle_data_structure.locally_owned_particle_range(),
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
}

/**
 * Verifies that the `reinit()` function correctly reinitializes the data structure and
 * redistributes particles across MPI processes. In this test, we insert particles from two
 * different MPI processes, call `reinit()`, and check that both the locally owned and ghost
 * particle range on each process matches the expected reference particles. The test ensures that
 * the particles are correctly redistributed and accessible on their respective owning ranks after
 * reinitialization.
 */
TEST_F(ObstacleDataStructureTest, Reinit)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  std::vector<ReferenceParticleProperties> references_locally_owned_particles;
  std::vector<ReferenceParticleProperties> references_ghost_particles;
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      references_locally_owned_particles = reference_particles_coarse_cell_0;
      for (auto &particle : reference_particles_coarse_cell_1)
        if (particle.is_ghost_on_other_mpi_process)
          references_ghost_particles.push_back(particle);
    }
  else
    {
      references_locally_owned_particles = reference_particles_coarse_cell_1;
      for (auto &particle : reference_particles_coarse_cell_0)
        if (particle.is_ghost_on_other_mpi_process)
          references_ghost_particles.push_back(particle);
    }

  {
    SCOPED_TRACE("Rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
                 ": Check for expected locally owned particles.");
    compare_particle_range_with_reference_particles(
      obstacle_data_structure.locally_owned_particle_range(), references_locally_owned_particles);
  }

  {
    SCOPED_TRACE("Rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
                 ": Check for expected ghost particles.");
    compare_particle_range_with_reference_particles(obstacle_data_structure.ghost_particle_range(),
                                                    references_ghost_particles);
  }
}

/**
 * As the test above, this test verifies that the `reinit()` function correctly reinitializes the
 * data structure and redistributes particles across MPI processes. However, in this test, we only
 * insert particles on one MPI process such that only on one process locally owned particles are
 * present. It is checked that the other MPI process correctly receives the ghost particles after
 * reinitialization but does not have any locally owned particles.
 */
TEST_F(ObstacleDataStructureTest, ReinitNoLocallyOwnedParticles)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    insert_particles(reference_particles_coarse_cell_0);
  else
    insert_particles({}); // Insert no particles on rank 1
  obstacle_data_structure.reinit();

  std::vector<ReferenceParticleProperties> references_locally_owned_particles;
  std::vector<ReferenceParticleProperties> references_ghost_particles;
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      references_locally_owned_particles = reference_particles_coarse_cell_0;
    }
  else
    {
      for (auto &particle : reference_particles_coarse_cell_0)
        if (particle.is_ghost_on_other_mpi_process)
          references_ghost_particles.push_back(particle);
    }

  {
    SCOPED_TRACE("Rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
                 ": Check for expected locally owned particles.");
    compare_particle_range_with_reference_particles(
      obstacle_data_structure.locally_owned_particle_range(), references_locally_owned_particles);
  }

  {
    SCOPED_TRACE("Rank " +
                 std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
                 ": Check for expected ghost particles.");
    compare_particle_range_with_reference_particles(obstacle_data_structure.ghost_particle_range(),
                                                    references_ghost_particles);
  }
}

/**
 * Verifies that the `find_particles_in_neighborhood()` function correctly identifies neighboring
 * particles within a specified relative tolerance. In this test, we select a particle of interest
 * and check that the function returns the expected neighboring particles based on their locations
 * and radii. The test compares the results with a reference set of neighboring particles to ensure
 * correctness.
 */
TEST_F(ObstacleDataStructureTest, FindParticlesInNeighborhood)
{
  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  constexpr double relative_tolerance = 0.15;

  const DEMParticleAccessor<dim, double> particle_of_interest =
    *obstacle_data_structure.locally_owned_particle_range().begin();

  auto neighbor_particles =
    obstacle_data_structure.find_particles_in_neighborhood(particle_of_interest,
                                                           relative_tolerance);

  std::vector<ReferenceParticleProperties> reference_neighbors;
  for (const auto &particle : reference_particles_coarse_cell_0)
    {
      if (particle.location == particle_of_interest.get_location() and
          particle.radius == particle_of_interest.radius())
        continue;

      const double distance = particle.location.distance(particle_of_interest.get_location());
      if (distance <= (1 + relative_tolerance) * (particle_of_interest.radius() + particle.radius))
        reference_neighbors.push_back(particle);
    }
  for (const auto &particle : reference_particles_coarse_cell_1)
    {
      if (particle.location == particle_of_interest.get_location() and
          particle.radius == particle_of_interest.radius())
        continue;

      const double distance = particle.location.distance(particle_of_interest.get_location());
      if (distance <= (1 + relative_tolerance) * (particle_of_interest.radius() + particle.radius))
        reference_neighbors.push_back(particle);
    }

  compare_particle_range_with_reference_particles(neighbor_particles, reference_neighbors);
}

/**
 * Verifies that the `find_particles_in_neighborhood()` function correctly identifies that there are
 * no neighboring particles for an isolated particle. In this test, we select a particle of interest
 * that is expected to have no neighbors within the specified relative tolerance and check that the
 * function returns an empty range.
 */
TEST_F(ObstacleDataStructureTest, FindParticlesInNeighborhoodIsolatedParticle)
{
  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  constexpr double relative_tolerance = 0.0;

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      // Search for the particle with location (0.7, 0.65, 0.6) in the locally owned particle range
      // for which we expect no contact neighbors to be found.
      const DEMParticleAccessor<dim, double> particle_of_interest =
        *std::find_if(obstacle_data_structure.locally_owned_particle_range().begin(),
                      obstacle_data_structure.locally_owned_particle_range().end(),
                      [](const DEMParticleAccessor<dim, double> &particle) {
                        return particle.get_location() == dealii::Point<dim>(0.7, 0.65, 0.6);
                      });

      auto neighbor_particles =
        obstacle_data_structure.find_particles_in_neighborhood(particle_of_interest,
                                                               relative_tolerance);

      EXPECT_TRUE(neighbor_particles.empty())
        << "Expected no neighboring particles for the isolated particle, but found "
        << neighbor_particles.size() << " neighbors.";
    }
}

/**
 * Verifies that the `update_ghost_particle_properties()` function correctly updates the properties
 * of ghost particles. In this test, we modify the properties of locally owned particles and then
 * call `update_ghost_particle_properties()` to ensure that the changes are correctly reflected in
 * the corresponding ghost particles on other MPI processes.
 */
TEST_F(ObstacleDataStructureTest, UpdateGhostParticleProperties)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  for (DEMParticleAccessor<dim, double> &particle :
       obstacle_data_structure.locally_owned_particle_range())
    {
      for (unsigned int d = 0; d < dim; ++d)
        {
          particle.linear_velocity(d) += 0.1 * (d + 1);
          particle.get_location()[d] += 0.2 * (d + 1);
        }
    }

  std::vector<ReferenceParticleProperties> references_ghost_particles;
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      for (auto &particle : reference_particles_coarse_cell_1)
        if (particle.is_ghost_on_other_mpi_process)
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                particle.velocity[d] += 0.1 * (d + 1);
                particle.location[d] += 0.2 * (d + 1);
              }
            references_ghost_particles.push_back(particle);
          }
    }
  else
    {
      for (auto &particle : reference_particles_coarse_cell_0)
        if (particle.is_ghost_on_other_mpi_process)
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                particle.velocity[d] += 0.1 * (d + 1);
                particle.location[d] += 0.2 * (d + 1);
              }
            references_ghost_particles.push_back(particle);
          }
    }
  obstacle_data_structure.update_ghost_particle_properties();
  compare_particle_range_with_reference_particles(obstacle_data_structure.ghost_particle_range(),
                                                  references_ghost_particles);
}

/**
 * Verifies that the `compress()` function accumulates the force/torque tensors on a ghost particle
 * back into the matching locally owned particle on its owning rank. For this test, we displace the
 * ghost particles on each rank, modify their force and torque, and then call `compress()` to ensure
 * that the changes are correctly reflected in the locally owned particles on the owning rank.
 */
TEST_F(ObstacleDataStructureTest, Compress)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  constexpr dealii::Tensor<1, dim, double>            force_increment({1.0, 2.0, 3.0});
  constexpr dealii::Tensor<1, axial_dim<dim>, double> torque_increment({4.0, 5.0, 6.0});
  for (DEMParticleAccessor<dim, double> &particle : obstacle_data_structure.ghost_particle_range())
    {
      particle.set_force(force_increment);
      particle.set_torque(torque_increment);
    }

  std::vector<ReferenceParticleProperties> references_compressed_particles;
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      references_compressed_particles = reference_particles_coarse_cell_0;
    }
  else
    {
      references_compressed_particles = reference_particles_coarse_cell_1;
    }

  for (auto &particle : references_compressed_particles)
    {
      if (particle.is_ghost_on_other_mpi_process)
        {
          particle.force += force_increment;
          particle.torque += torque_increment;
        }
    }
  obstacle_data_structure.compress();
  compare_particle_range_with_reference_particles(
    obstacle_data_structure.locally_owned_particle_range(), references_compressed_particles);
}

/**
 * This test checks that the `get_obstacles_in_cell` function correctly retrieves only locally owned
 * particles when the specified cell does not have any adjacent cells on other MPI processes. It
 * ensures that the function returns only the particles that are owned by the current MPI process.
 *
 * The test uses a specific cell ID of an active cell for which the parent cell on the level where
 * all adjacent cells do not have any ghost particles. This setup guarantees that the neighboring
 * particles consist solely of locally owned particles.
 */
TEST_F(ObstacleDataStructureTest, GetObstaclesInCellOnlyLocallyOwned)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      // Use a cell id of an active cell which parent cell on the level where particles are cached
      // does not have any adjacent cells on the other MPI process. This ensures that only locally
      // owned particles are returned, as ghost particles are only available for adjacent cells on
      // other MPI processes.
      dealii::TriaIterator<dealii::CellAccessor<dim>> cell =
        triangulation.create_cell_iterator(dealii::CellId("0_4:2513"));

      std::vector<ReferenceParticleProperties> reference_relevant_cell_particles =
        reference_particles_coarse_cell_0;

      std::vector<DEMParticleAccessor<dim, double>> relevant_cell_particles;
      obstacle_data_structure.get_obstacles_in_cell(cell, relevant_cell_particles);
      compare_particle_range_with_reference_particles(relevant_cell_particles,
                                                      reference_relevant_cell_particles);
    }
}

/**
 * This test checks that the `get_obstacles_in_cell` function correctly retrieves both locally owned
 * and ghost particles when the specified cell has adjacent cells on other MPI processes. It ensures
 * that the function returns all relevant particles, including those that are ghost particles on the
 * current MPI process.
 *
 * The test uses a specific cell ID of an active cell for which the parent cell on the level where
 * particles are cached has adjacent cells on both the current and other MPI processes. This setup
 * guarantees that the neighboring particles consist of both locally owned and ghost particles.
 */
TEST_F(ObstacleDataStructureTest, GetObstaclesInCellLocallyOwnedAndGhost)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      // Use a cell id of an active cell which parent cell on the level where particles are cached
      // has adjacent cells on the other MPI process and on the own MPI process. This ensures that
      // the neighboring particles consist of both locally owned and ghost particles, as ghost
      // particles are only available for adjacent cells on other MPI processes.
      dealii::TriaIterator<dealii::CellAccessor<dim>> cell =
        triangulation.create_cell_iterator(dealii::CellId("0_4:3253"));

      std::vector<ReferenceParticleProperties> reference_relevant_cell_particles =
        reference_particles_coarse_cell_0;

      for (const auto &particle : reference_particles_coarse_cell_1)
        {
          if (particle.location[0] < 1.5)
            reference_relevant_cell_particles.push_back(particle);
        }

      std::vector<DEMParticleAccessor<dim, double>> relevant_cell_particles;
      obstacle_data_structure.get_obstacles_in_cell(cell, relevant_cell_particles);
      compare_particle_range_with_reference_particles(relevant_cell_particles,
                                                      reference_relevant_cell_particles);
    }
}

/**
 * Verifies that the get_obstacles_in_cell function returns an empty range when no particles are
 * present in the parent cell on the storage level and its adjacent cells.
 */
TEST_F(ObstacleDataStructureTest, GetObstaclesInCellEmpty)
{
  constexpr unsigned int required_mpi_processes = 2;
  TestUtils::skip_if_not_correct_mpi_process_count<dim, double>(required_mpi_processes);

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    insert_particles({});
  else
    insert_particles(reference_particles_coarse_cell_1);
  obstacle_data_structure.reinit();

  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      // Use a cell id of an active cell which parent cell on the level where particles are cached
      // does not have any adjacent cells on the other MPI process. This ensures that only locally
      // owned particles are returned, as ghost particles are only available for adjacent cells on
      // other MPI processes. As no locally owned particles are inserted on this rank, the expected
      // result is an empty range.
      dealii::TriaIterator<dealii::CellAccessor<dim>> cell =
        triangulation.create_cell_iterator(dealii::CellId("0_4:2513"));

      std::vector<DEMParticleAccessor<dim, double>> relevant_cell_particles;
      obstacle_data_structure.get_obstacles_in_cell(cell, relevant_cell_particles);
      EXPECT_TRUE(relevant_cell_particles.empty())
        << "Expected no particles in the specified cell, but found "
        << relevant_cell_particles.size() << " particles.";
    }
}

/**
 * Verifies that the NeighborListUpdateTracker correctly identifies that an update is not required
 * when all particles are displaced within the specified maximum displacement and skin thickness. In
 * this test, we displace two particles in the locally owned range such that they remain within the
 * allowed limits, and we expect the tracker to indicate that no update is required.
 */
TEST_F(ObstacleDataStructureTest, NeighborListUpdateTrackerNoUpdateRequired)
{
  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  NeighborListUpdateTracker<dim, double, SphericalParticle<dim, double>> tracker(
    obstacle_data_structure);
  tracker.reinit_after_update(0.2, 0.1);

  for (DEMParticleAccessor<dim, double> &particle :
       obstacle_data_structure.locally_owned_particle_range())
    {
      if (particle.local_id() < 2)
        {
          particle.get_location() += dealii::Point<dim, double>({0.04, 0.013, 0.02});
        }
    }

  EXPECT_FALSE(tracker.update_required());
}

/**
 * Verifies that the NeighborListUpdateTracker correctly identifies when an update is required due
 * to particles moving beyond the specified maximum displacement. In this test, we displace two
 * particles such that they exceed the maximum displacement, and we expect the tracker to indicate
 * that an update is required.
 *
 * @note The skin thickness is set to a very large value to ensure that the update requirement is
 * solely determined by the maximum displacement.
 */
TEST_F(ObstacleDataStructureTest, NeighborListUpdateTrackerUpdateRequiredMaxDisplacement)
{
  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  NeighborListUpdateTracker<dim, double, SphericalParticle<dim, double>> tracker(
    obstacle_data_structure);
  tracker.reinit_after_update(std::numeric_limits<double>::max(), 0.1);

  for (DEMParticleAccessor<dim, double> &particle :
       obstacle_data_structure.locally_owned_particle_range())
    {
      if (particle.local_id() < 2)
        {
          particle.get_location() += dealii::Point<dim, double>({0.1, 0.013, 0.02});
        }
    }

  EXPECT_TRUE(tracker.update_required());
}

/**
 * Verifies that the NeighborListUpdateTracker correctly identifies when an update is required due
 * to particles moving beyond the specified skin thickness. In this test, we displace two particles
 * in the locally owned range such that they exceed the skin thickness, and we expect the tracker to
 * indicate that an update is required on all MPI processes, even though only rank 0 displaces the
 * particles.
 *
 * @note The maximum per particle displacement is set to a very large value to ensure that the
 * update requirement is solely determined by the skin thickness.
 */
TEST_F(ObstacleDataStructureTest, NeighborListUpdateTrackerUpdateRequiredSkinThickness)
{
  insert_particles(
    reference_particles_on_rank[dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)]);
  obstacle_data_structure.reinit();

  NeighborListUpdateTracker<dim, double, SphericalParticle<dim, double>> tracker(
    obstacle_data_structure);
  tracker.reinit_after_update(0.1, std::numeric_limits<double>::max());

  for (DEMParticleAccessor<dim, double> &particle :
       obstacle_data_structure.locally_owned_particle_range())
    {
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          if (particle.local_id() == 0)
            {
              particle.get_location() += dealii::Point<dim, double>({0.05, 0.0, 0.0});
            }
          else if (particle.local_id() == 1)
            {
              particle.get_location() += dealii::Point<dim, double>({0.051, 0.0, 0.0});
            }
        }
    }

  // Only rank 0 displaces particles; update_required() is expected to reduce across all ranks so
  // that every rank returns the same value.
  EXPECT_TRUE(tracker.update_required());
}
