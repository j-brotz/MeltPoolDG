/**
 * @file Benchmarks for evaluating the performance of particle-to-particle and particle-to-wall
 * contact force calculations within MeltPoolDG. The benchmarks asses the performance of the contact
 * force computation using the function add_load_to_obstacles() of the SphericalParticleContactForce
 * class for different numbers of particles and contact configurations.
 */

#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/contact_forces_data.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>
#include <meltpooldg/time_integration/time_stepping_data.hpp>

#include <benchmark/benchmark.h>
#include <benchmark_util.hpp>

#include <memory>
#include <vector>

namespace
{
  constexpr int dim = 3;

  using number = double;
  using namespace MeltPoolDG;

  /**
   * Fixture for benchmarking particle contact forces. This fixture sets up a simple triangulation
   * and initializes the obstacle field and contact force objects.
   */
  class ParticleContactForceFixture : public benchmark::Fixture
  {
    using ObstacleType = SphericalParticle<dim, number>;

  public:
    struct Data
    {
      /**
       * Constructor which initializes the triangulation, obstacle field, and contact force objects.
       * The triangulation consists of a 3-dimensional domain of size 2x2x2. The obstacle field is
       * initialized without any particles inserted.
       */
      Data()
        : triangulation(MPI_COMM_WORLD,
                        dealii::Triangulation<dim>::MeshSmoothing::none,
                        dealii::parallel::distributed::Triangulation<
                          dim>::Settings::construct_multigrid_hierarchy)
        , time_iterator(time_stepping_data)
      {
        // Setup the triangulation with a simple rectangular domain
        std::vector<unsigned int> n_base_cells{2, 2, 2};
        dealii::Point<dim>        bottom_left(0, 0, 0);
        dealii::Point<dim>        top_right(2, 2, 2);

        dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          n_base_cells,
                                                          bottom_left,
                                                          top_right);

        triangulation.refine_global(4);

        // Setup the particle field
        std::vector<dealii::Point<dim, number>> dummy_locations;
        std::vector<std::vector<number>>        dummy_properties;
        obstacle_field = std::make_unique<ObstacleField<dim, number, ObstacleType>>(
          obstacle_data, triangulation, mapping, dummy_locations, dummy_properties);

        // Setup the contact force object
        contact_force =
          std::make_unique<SphericalParticleContactForce<dim, number, ObstacleType>>(contact_data,
                                                                                     time_iterator);
      }

      /**
       * A helper function to insert particles into the obstacle field. The function takes vectors
       * of particle locations and corresponding particle radii, constructs the required properties
       * vector, and calls the insert_obstacles() method of the obstacle field to add the particles.
       *
       * @param particle_locations A vector of points representing the locations of the particles
       * to be inserted.
       * @param particle_radii A vector of numbers representing the radii of the particles to be
       * inserted. The size of this vector must match the size of the particle_locations vector.
       */
      void
      insert_particles(const std::vector<dealii::Point<dim, number>> &particle_locations,
                       const std::vector<number>                     &particle_radii)
      {
        Assert(particle_locations.size() == particle_radii.size(),
               dealii::ExcMessage(
                 "The number of particle locations must match the number of particle radii."));

        std::vector<std::vector<number>> particle_properties = {};
        particle_properties.reserve(particle_locations.size());
        for (unsigned int i = 0; i < particle_locations.size(); ++i)
          {
            std::vector<number> properties(ObstacleType::n_obstacle_properties, 0);
            properties[ObstacleType::Properties::radius] = particle_radii[i];

            particle_properties.push_back(properties);
          }

        obstacle_field->insert_obstacles(triangulation, particle_locations, particle_properties);
      }

      dealii::parallel::distributed::Triangulation<dim>                         triangulation;
      dealii::MappingQ1<dim>                                                    mapping;
      ObstacleData<number>                                                      obstacle_data;
      std::unique_ptr<ObstacleField<dim, number, ObstacleType>>                 obstacle_field;
      SphericalParticleContactData<number>                                      contact_data;
      std::unique_ptr<SphericalParticleContactForce<dim, number, ObstacleType>> contact_force;
      TimeIntegration::TimeSteppingData<number>                                 time_stepping_data;
      TimeIntegration::TimeIterator<number>                                     time_iterator;
    };

    // Prevent compiler warnings about hiding overloaded virtual functions (-Woverloaded-virtual).
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

    /**
     * Fixture setup function called before each benchmark run. Initializes the
     * benchmark-specific data structure using the benchmark state.
     */
    void
    SetUp(benchmark::State &) override
    {
      data = std::make_unique<Data>();
    }

    /**
     * Fixture teardown function called after each benchmark run. Cleans up and deallocates
     * the benchmark-specific data, i.e., the data object struct.
     */
    void
    TearDown(benchmark::State &) override
    {
      data.reset();
    }

    std::unique_ptr<Data> data;
  };

  /**
   * A benchmark to measure the performance of the contact force calculation for a single isolated
   * particle-particle contact. Only two particles which are in contact with each other are placed
   * in the domain, without any additional particles or walls.
   */
  BENCHMARK_DEFINE_F(ParticleContactForceFixture, SingleParticleParticleContact)
  (benchmark::State &state)
  {
    std::vector<dealii::Point<dim, number>> particle_locations = {
      dealii::Point<dim, number>(1, 1, 1), dealii::Point<dim, number>(1.45, 1, 1)};
    std::vector<number> particle_radii = {0.2, 0.3};

    this->data->insert_particles(particle_locations, particle_radii);

    mpi_benchmark_loop(state, [&]() {
      this->data->contact_force->add_load_to_obstacles(*this->data->obstacle_field);
    });
    state.counters["Particles"] = particle_locations.size();
  }

  /**
   * A benchmark to measure the performance of the contact force calculation for a single isolated
   * particle-wall contact. Only one particle which is in contact with a single wall is placed in
   * the domain, without any additional particles or walls.
   */
  BENCHMARK_DEFINE_F(ParticleContactForceFixture, SingleParticleWallContact)
  (benchmark::State &state)
  {
    std::vector<dealii::Point<dim, number>> particle_locations = {
      dealii::Point<dim, number>(1, 1, 0.4)};
    std::vector<number> particle_radii = {0.5};

    this->data->insert_particles(particle_locations, particle_radii);

    dealii::Tensor<1, dim, number> ground_wall_normal;
    ground_wall_normal[dim - 1] = 1.;
    this->data->contact_force->attach_wall(
      std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(dealii::Point<dim>(),
                                                                      ground_wall_normal));

    mpi_benchmark_loop(state, [&]() {
      this->data->contact_force->add_load_to_obstacles(*this->data->obstacle_field);
    });
    state.counters["Particles"] = particle_locations.size();
  }

  /**
   * A benchmark to measure the performance of the contact force calculation for multiple
   * particle-particle contacts. A grid of particles is placed in the domain, with each particle in
   * contact with its neighbors. No walls are present in this benchmark.
   */
  BENCHMARK_DEFINE_F(ParticleContactForceFixture, MultipleParticleParticleContact)
  (benchmark::State &state)
  {
    constexpr unsigned int n_rows      = 5;
    constexpr unsigned int n_particles = n_rows * n_rows;

    number offset = 2.0 / (n_rows + 1);

    std::vector<dealii::Point<dim, number>> particle_locations;
    particle_locations.reserve(n_particles);
    for (unsigned int i = 0; i < n_rows; ++i)
      {
        for (unsigned int j = 0; j < n_rows; ++j)
          {
            dealii::Point<dim, number> location((i + 1) * offset, (j + 1) * offset, 0.4);
            particle_locations.push_back(location);
          }
      }
    std::vector<number> particle_radii(n_particles, 0.2);

    this->data->insert_particles(particle_locations, particle_radii);

    mpi_benchmark_loop(state, [&]() {
      this->data->contact_force->add_load_to_obstacles(*this->data->obstacle_field);
    });
    state.counters["Particles"] = particle_locations.size();
  }

  /**
   * A benchmark to measure the performance of the contact force calculation for multiple
   * particle-wall contacts. A grid of particles is placed in the domain, with each particle in
   * contact with a single wall. No particle-particle contacts are present in this benchmark.
   */
  BENCHMARK_DEFINE_F(ParticleContactForceFixture, MultipleParticleWallContact)
  (benchmark::State &state)
  {
    constexpr unsigned int n_rows      = 5;
    constexpr unsigned int n_particles = n_rows * n_rows;

    number offset = 2.0 / (n_rows + 1);

    std::vector<dealii::Point<dim, number>> particle_locations;
    particle_locations.reserve(n_particles);
    for (unsigned int i = 0; i < n_rows; ++i)
      {
        for (unsigned int j = 0; j < n_rows; ++j)
          {
            dealii::Point<dim, number> location((i + 1) * offset, (j + 1) * offset, 0.4);
            particle_locations.push_back(location);
          }
      }
    std::vector<number> particle_radii(n_particles, 0.1);

    this->data->insert_particles(particle_locations, particle_radii);

    dealii::Tensor<1, dim, number> ground_wall_normal;
    ground_wall_normal[dim - 1] = 1.;
    this->data->contact_force->attach_wall(
      std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(dealii::Point<dim>(),
                                                                      ground_wall_normal));

    mpi_benchmark_loop(state, [&]() {
      this->data->contact_force->add_load_to_obstacles(*this->data->obstacle_field);
    });
    state.counters["Particles"] = particle_locations.size();
  }
} // namespace

BENCHMARK_REGISTER_F(ParticleContactForceFixture, SingleParticleParticleContact)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(ParticleContactForceFixture, SingleParticleWallContact)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(ParticleContactForceFixture, MultipleParticleParticleContact)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK_REGISTER_F(ParticleContactForceFixture, MultipleParticleWallContact)
  ->Unit(benchmark::kMicrosecond);

MPDG_BENCHMARK_MPI_MAIN;
