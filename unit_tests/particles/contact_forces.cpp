
#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <memory>
#include <numbers>

namespace
{
  constexpr int dim = 3;

  struct ParticleTestData
  {
    int                            id;
    double                         radius;
    double                         density;
    dealii::Tensor<1, dim, double> linear_velocity;
    dealii::Point<dim, double>     location;
  };

  class ContactForceFixture : public ::testing::Test
  {
  protected:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

    ContactForceFixture()
      : mapping(1)
      , contact_data({.restitution_coefficient = 0.4,
                      .particle = {.youngs_modulus = 26.25e6, .poisson_ratio = 0.342}})
      , contact_force(contact_data)
    {}

    void
    SetUp() override
    {
      dealii::GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(4);

      std::vector<std::vector<double>> particle_properties = {};
      std::vector<dealii::Point<dim>>  particle_locations  = {};

      obstacle_field = std::make_unique<MeltPoolDG::ObstacleField<dim, double, ObstacleType>>(
        obstacle_data, triangulation, mapping, particle_locations, particle_properties);
    }

    void
    insert_particle(const ParticleTestData &particle_data)
    {
      std::vector<double> properties(ObstacleType::n_obstacle_properties, 0);
      for (int d = 0; d < dim; ++d)
        properties[ObstacleType::Properties::velocity + d] = particle_data.linear_velocity[d];
      properties[ObstacleType::Properties::radius]  = particle_data.radius;
      properties[ObstacleType::Properties::density] = particle_data.density;
      properties[ObstacleType::Properties::mass] =
        dim == 3 ?
          (4.0 / 3.0) * std::numbers::pi * particle_data.radius * particle_data.radius *
            particle_data.radius * particle_data.density :
          std::numbers::pi * particle_data.radius * particle_data.radius * particle_data.density;
      properties[ObstacleType::Properties::particle_id] = particle_data.id;

      std::vector<dealii::Point<dim>>  particle_locations  = {particle_data.location};
      std::vector<std::vector<double>> particle_properties = {properties};

      obstacle_field->insert_obstacles(triangulation, particle_locations, particle_properties);
    }

    dealii::Triangulation<dim>                                            triangulation;
    dealii::MappingQ<dim>                                                 mapping;
    MeltPoolDG::ObstacleData<double>                                      obstacle_data;
    std::unique_ptr<MeltPoolDG::ObstacleField<dim, double, ObstacleType>> obstacle_field;
    MeltPoolDG::SphericalParticleContactData<double>                      contact_data;
    MeltPoolDG::SphericalParticleContactForce<dim, double, ObstacleType>  contact_force;
  };

  TEST_F(ContactForceFixture, NormalParticleParticleContactSameRadiusSameDensity)
  {
    using particle_id_type = int;

    insert_particle({.id              = 0,
                     .radius          = 0.2,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.05, 0, 0}),
                     .location        = dealii::Point<dim>(0.4, 0.5, 0.5)});
    insert_particle({.id              = 1,
                     .radius          = 0.2,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.02, 0, 0}),
                     .location        = dealii::Point<dim>(0.79, 0.5, 0.5)});

    contact_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<particle_id_type, dealii::Tensor<1, dim, double>> expected_forces_map = {
      {0, dealii::Tensor<1, dim, double>({-6393.2109, 0.0, 0.0})},
      {1, dealii::Tensor<1, dim, double>({6393.2109, 0.0, 0.0})}};

    for (const auto &particle : obstacle_field->locally_owned_particle_range())
      {
        const particle_id_type particle_id = static_cast<particle_id_type>(particle.id());
        const dealii::Tensor<1, dim, double> computed_force = particle.get_force();
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }

  TEST_F(ContactForceFixture, NormalParticleParticleContactSameRadiusDifferentDensity)
  {
    using particle_id_type = int;

    insert_particle({.id              = 0,
                     .radius          = 0.2,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.05, 0, 0}),
                     .location        = dealii::Point<dim>(0.4, 0.5, 0.5)});
    insert_particle({.id              = 1,
                     .radius          = 0.2,
                     .density         = 2700,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.02, 0, 0}),
                     .location        = dealii::Point<dim>(0.79, 0.5, 0.5)});

    contact_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<particle_id_type, dealii::Tensor<1, dim, double>> expected_forces_map = {
      {0, dealii::Tensor<1, dim, double>({-6377.8522, 0.0, 0.0})},
      {1, dealii::Tensor<1, dim, double>({6377.8522, 0.0, 0.0})}};

    for (const auto &particle : obstacle_field->locally_owned_particle_range())
      {
        const particle_id_type particle_id = static_cast<particle_id_type>(particle.id());
        const dealii::Tensor<1, dim, double> computed_force = particle.get_force();
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }

  TEST_F(ContactForceFixture, NormalParticleParticleContactDifferentRadius)
  {
    using particle_id_type = int;

    insert_particle({.id              = 0,
                     .radius          = 0.2,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.05, 0, 0}),
                     .location        = dealii::Point<dim>(0.4, 0.5, 0.5)});
    insert_particle({.id              = 1,
                     .radius          = 0.1,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.02, 0, 0}),
                     .location        = dealii::Point<dim>(0.69, 0.5, 0.5)});

    contact_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<particle_id_type, dealii::Tensor<1, dim, double>> expected_forces_map = {
      {0, dealii::Tensor<1, dim, double>({-5170.7448, 0.0, 0.0})},
      {1, dealii::Tensor<1, dim, double>({5170.7448, 0.0, 0.0})}};

    for (const auto &particle : obstacle_field->locally_owned_particle_range())
      {
        const particle_id_type particle_id = static_cast<particle_id_type>(particle.id());
        const dealii::Tensor<1, dim, double> computed_force = particle.get_force();
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }

  TEST_F(ContactForceFixture, NormalParticleParticleAngledContact)
  {
    using particle_id_type = int;

    insert_particle({.id              = 0,
                     .radius          = 0.2,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.02586, 0.04279, 0}),
                     .location        = dealii::Point<dim>(0.4, 0.5, 0.5)});
    insert_particle({.id              = 1,
                     .radius          = 0.1,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.01034, 0.01712, 0}),
                     .location        = dealii::Point<dim>(0.55, 0.7482, 0.5)});

    contact_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<particle_id_type, dealii::Tensor<1, dim, double>> expected_forces_map = {
      {0, dealii::Tensor<1, dim, double>({-2672.2475, -4421.6790, 0.0})},
      {1, dealii::Tensor<1, dim, double>({2672.2475, 4421.6790, 0.0})}};

    for (const auto &particle : obstacle_field->locally_owned_particle_range())
      {
        const particle_id_type particle_id = static_cast<particle_id_type>(particle.id());
        const dealii::Tensor<1, dim, double> computed_force = particle.get_force();
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }

  TEST_F(ContactForceFixture, NormalMultipleParticleParticleContact)
  {
    using particle_id_type = int;

    insert_particle({.id              = 0,
                     .radius          = 0.2,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.02, 0, 0}),
                     .location        = dealii::Point<dim>(0.11, 0.5, 0.5)});
    insert_particle({.id              = 1,
                     .radius          = 0.2,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.05, 0, 0}),
                     .location        = dealii::Point<dim>(0.5, 0.5, 0.5)});
    insert_particle({.id              = 2,
                     .radius          = 0.1,
                     .density         = 4300,
                     .linear_velocity = dealii::Tensor<1, dim, double>({0.02, 0, 0}),
                     .location        = dealii::Point<dim>(0.79, 0.5, 0.5)});

    contact_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<particle_id_type, dealii::Tensor<1, dim, double>> expected_forces_map = {
      {0, dealii::Tensor<1, dim, double>({-6140.7875, 0.0, 0.0})},
      {1, dealii::Tensor<1, dim, double>({970.0427, 0.0, 0.0})},
      {2, dealii::Tensor<1, dim, double>({5170.7448, 0.0, 0.0})}};

    for (const auto &particle : obstacle_field->locally_owned_particle_range())
      {
        const particle_id_type particle_id = static_cast<particle_id_type>(particle.id());
        const dealii::Tensor<1, dim, double> computed_force = particle.get_force();
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }
} // namespace
