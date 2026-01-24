
#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <limits>
#include <memory>
#include <numbers>

/**
 * This test suite verifies the correct implementation of contact forces between particles. In
 * particular, it validates the contact normal forces computed using the Hertzian contact model for
 * spherical particles, as described in, e.g., Golshan et al., “Lethe-DEM: An Open-Source Parallel
 * Discrete Element Solver with Load Balancing,” (https://doi.org/10.1007/s40571-022-00478-6).
 *
 * To this end, several particle configurations are considered, and the resulting contact forces are
 * compared against precomputed reference values. See the individual tests for further details.
 */
namespace
{
  constexpr int dim = 3;

  struct ParticleTestData
  {
    using particle_id_type = int;

    particle_id_type               id{};
    double                         radius{};
    double                         density{};
    dealii::Tensor<1, dim, double> angular_velocity{};
    dealii::Tensor<1, dim, double> linear_velocity{};
    dealii::Point<dim, double>     location{};
  };

  class ContactForceFixture : public ::testing::Test
  {
  protected:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

    ContactForceFixture()
      : mapping(1)
      , contact_data({.restitution_coefficient      = 0.4,
                      .sliding_friction_coefficient = 0.2,
                      .particle = {.youngs_modulus = 26.25e6, .poisson_ratio = 0.342}})
      , time_iterator(MeltPoolDG::TimeIntegration::TimeSteppingData<double>{.start_time     = 0.0,
                                                                            .end_time       = 1.0,
                                                                            .time_step_size = 1e-4})
      , contact_force(contact_data, time_iterator)
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
      for (int d = 0; d < MeltPoolDG::axial_dim<dim>; ++d)
        properties[ObstacleType::Properties::angular_velocity + d] =
          particle_data.angular_velocity[d];
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
    MeltPoolDG::TimeIntegration::TimeIterator<double>                     time_iterator;
    MeltPoolDG::SphericalParticleContactForce<dim, double, ObstacleType>  contact_force;
  };

  TEST_F(ContactForceFixture, ParticleParticleContact_SameRadius_DifferentDensity_PureNormalContact)
  {
    // This test verifies particle–particle contact behavior under the following conditions:
    // - Particles have the same radius.
    // - Particles have different densities.
    // - Particles move along the line of centers, producing a purely normal contact force.
    // - No tangential relative motion is present, so tangential forces are expected to be zero.
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

    const std::map<ParticleTestData::particle_id_type, dealii::Tensor<1, dim, double>>
      expected_forces_map = {{0, dealii::Tensor<1, dim, double>({-6377.8522, 0.0, 0.0})},
                             {1, dealii::Tensor<1, dim, double>({6377.8522, 0.0, 0.0})}};

    for (const auto &particle : this->obstacle_field->get_particle_handler())
      {
        const ParticleTestData::particle_id_type particle_id =
          static_cast<ParticleTestData::particle_id_type>(
            ObstacleType::get_property(particle, ObstacleType::Properties::particle_id));
        const dealii::Tensor<1, dim, double> computed_force = ObstacleType::get_force(particle);
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }

  TEST_F(ContactForceFixture, ParticleParticleContact_SingleAngularVelocity_NoCoulombLimit)
  {
    // This test verifies particle–particle contact for both, normal and tangential forces under the
    // following conditions:
    // - Particles have the same radius and density.
    // - Particles have different normal velocities.
    // - Only one particle has a non-zero angular velocity, creating tangential relative motion.
    // - Tangential contact forces arise solely from rotational motion of a single particle.
    // - The Coulomb friction limit is disabled by setting the friction coefficient to infinity.
    // - A single time step is used
    insert_particle({.id               = 0,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 0, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({0.05, 0, 0}),
                     .location         = dealii::Point<dim>(0.4, 0.5, 0.5)});
    insert_particle({.id               = 1,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 10, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({0.02, 0, 0}),
                     .location         = dealii::Point<dim>(0.79, 0.5, 0.5)});

    contact_data.sliding_friction_coefficient = std::numeric_limits<double>::infinity();

    contact_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<ParticleTestData::particle_id_type, dealii::Tensor<1, dim, double>>
      expected_forces_map = {{0, dealii::Tensor<1, dim, double>({-6393.2109, 0.0, 7645.4767})},
                             {1, dealii::Tensor<1, dim, double>({6393.2109, 0.0, -7645.4767})}};

    for (const auto &particle : this->obstacle_field->get_particle_handler())
      {
        const ParticleTestData::particle_id_type particle_id =
          static_cast<ParticleTestData::particle_id_type>(
            ObstacleType::get_property(particle, ObstacleType::Properties::particle_id));
        const dealii::Tensor<1, dim, double> computed_force = ObstacleType::get_force(particle);
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }

  TEST_F(ContactForceFixture,
         ParticleParticleContact_DifferentRadii_DifferentAngularVelocities_NoCoulombLimit)
  {
    // This test verifies particle–particle contact behavior under the following conditions:
    // - Particles have different radii, affecting both normal and tangential contact forces.
    // - Particles have different angular velocities, influencing tangential contact forces.
    // - A single time step is used.
    // - The Coulomb friction limit is disabled by setting the friction coefficient to infinity.
    insert_particle({.id               = 0,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, -3, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({0.05, 0, 0}),
                     .location         = dealii::Point<dim>(0.4, 0.5, 0.5)});
    insert_particle({.id               = 1,
                     .radius           = 0.1,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 10, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({0.02, 0, 0}),
                     .location         = dealii::Point<dim>(0.69, 0.5, 0.5)});

    contact_data.sliding_friction_coefficient = std::numeric_limits<double>::infinity();

    contact_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<ParticleTestData::particle_id_type, dealii::Tensor<1, dim, double>>
      expected_forces_map = {{0, dealii::Tensor<1, dim, double>({-5170.7448, 0.0, 662.9924})},
                             {1, dealii::Tensor<1, dim, double>({5170.7448, 0.0, -662.9924})}};

    for (const auto &particle : this->obstacle_field->get_particle_handler())
      {
        const ParticleTestData::particle_id_type particle_id =
          static_cast<ParticleTestData::particle_id_type>(
            ObstacleType::get_property(particle, ObstacleType::Properties::particle_id));
        const dealii::Tensor<1, dim, double> computed_force = ObstacleType::get_force(particle);
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }

  TEST_F(ContactForceFixture,
         ParticleParticleContact_MultipleSimultaneousContacts_NoCoulombLimit_MultiStep)
  {
    // This test verifies particle–particle contact behavior under the following conditions:
    // - Three particles are in simultaneous contact, creating multiple overlapping interactions.
    // - Particles have identical radii and material properties.
    // - Relative motion includes both translational and rotational components.
    // - Tangential contact forces are unbounded because the Coulomb friction limit is disabled.
    insert_particle({.id               = 0,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 0, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({-5, 0, 0.05}),
                     .location         = dealii::Point<dim>(0.5, 0.5, 0.5)});
    insert_particle({.id               = 1,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 0, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({6, 0, 0.02}),
                     .location         = dealii::Point<dim>(0.5, 0.5, 0.89)});
    insert_particle({.id               = 2,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 15, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({0, 0, 0.02}),
                     .location         = dealii::Point<dim>(0.5, 0.5, 0.11)});

    contact_data.sliding_friction_coefficient = std::numeric_limits<double>::infinity();

    time_iterator.set_current_time_increment(1e-4);

    const std::array<std::map<ParticleTestData::particle_id_type, dealii::Tensor<1, dim, double>>,
                     3>
      expected_forces_map = {
        {{{0, dealii::Tensor<1, dim, double>({72632.0283, 0.0, -252.4233})},
          {1, dealii::Tensor<1, dim, double>({-42050.1217, 0.0, 6393.2109})},
          {2, dealii::Tensor<1, dim, double>({-30581.9067, 0.0, -6140.7875})}},
         {{0, dealii::Tensor<1, dim, double>({74049.7007, 0.0, -252.4233})},
          {1, dealii::Tensor<1, dim, double>({-42870.8793, 0.0, 6393.2109})},
          {2, dealii::Tensor<1, dim, double>({-31178.8213, 0.0, -6140.7875})}},
         {{0, dealii::Tensor<1, dim, double>({75467.3730, 0.0, -252.4233})},
          {1, dealii::Tensor<1, dim, double>({-43691.6370, 0.0, 6393.2109})},
          {2, dealii::Tensor<1, dim, double>({-31775.7359, 0.0, -6140.7875})}}}};

    for (const auto &expected_forces : expected_forces_map)
      {
        for (auto &particle : obstacle_field->locally_owned_particle_range())
          {
            // Reset forces to zero before each iteration
            particle.set_force(dealii::Tensor<1, dim, double>({0, 0, 0}));
          }

        contact_force.add_load_to_obstacles(*this->obstacle_field);

        for (const auto &particle : this->obstacle_field->get_particle_handler())
          {
            const ParticleTestData::particle_id_type particle_id =
              static_cast<ParticleTestData::particle_id_type>(
                ObstacleType::get_property(particle, ObstacleType::Properties::particle_id));
            const dealii::Tensor<1, dim, double> computed_force = ObstacleType::get_force(particle);
            const dealii::Tensor<1, dim, double> expected_force = expected_forces.at(particle_id);

            for (int d = 0; d < dim; ++d)
              {
                EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
                  << "Particle id: " << particle_id << ", force component: " << d;
              }
          }
      }
  }

  TEST_F(ContactForceFixture, ParticleParticleContact_WithSliding_CoulombLimitActive_MultiStep)
  {
    // This test verifies particle–particle contact behavior under the following conditions:
    // - Particles have identical radii and material properties.
    // - Relative motion includes a tangential component, inducing sliding at the contact.
    // - The Coulomb friction limit is enabled via a finite friction coefficient.
    // - Tangential forces are capped by the Coulomb limit when sliding occurs.
    // - Multiple consecutive contact evaluations are performed using a fixed time step.
    insert_particle({.id               = 0,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 0, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({-1.5, 0, 2}),
                     .location         = dealii::Point<dim>(0.5, 0.5, 0.3)});
    insert_particle({.id               = 1,
                     .radius           = 0.2,
                     .density          = 4300,
                     .angular_velocity = dealii::Tensor<1, dim, double>({0, 0, 0}),
                     .linear_velocity  = dealii::Tensor<1, dim, double>({0.5, 0, -1}),
                     .location         = dealii::Point<dim>(0.5, 0.5, 0.69)});

    contact_data.sliding_friction_coefficient = 0.406;

    time_iterator.set_current_time_increment(1e-4);

    const std::array<std::map<ParticleTestData::particle_id_type, dealii::Tensor<1, dim, double>>,
                     2>
      expected_forces_map = {
        {{{0, dealii::Tensor<1, dim, double>({7645.4767, 0.0, -18888.1665})},
          {1, dealii::Tensor<1, dim, double>({-7645.4767, 0.0, 18888.1665})}},
         {{0, dealii::Tensor<1, dim, double>({7668.5956, 0.0, -18888.1665})},
          {1, dealii::Tensor<1, dim, double>({-7668.5956, 0.0, 18888.1665})}}}};

    for (const auto &expected_forces : expected_forces_map)
      {
        for (auto &particle : obstacle_field->locally_owned_particle_range())
          {
            // Reset forces to zero before each iteration
            particle.set_force(dealii::Tensor<1, dim, double>({0, 0, 0}));
          }

        contact_force.add_load_to_obstacles(*this->obstacle_field);

        for (const auto &particle : this->obstacle_field->get_particle_handler())
          {
            const ParticleTestData::particle_id_type particle_id =
              static_cast<ParticleTestData::particle_id_type>(
                ObstacleType::get_property(particle, ObstacleType::Properties::particle_id));
            const dealii::Tensor<1, dim, double> computed_force = ObstacleType::get_force(particle);
            const dealii::Tensor<1, dim, double> expected_force = expected_forces.at(particle_id);

            for (int d = 0; d < dim; ++d)
              {
                EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
                  << "Particle id: " << particle_id;
              }
          }
      }
  }

  TEST_F(ContactForceFixture, ParticleParticleContact_DifferentRadii_ObliqueApproach)
  {
    // This test verifies particle–particle contact behavior under the following conditions:
    // - Particles have different radii.
    // - Particles approach each other along an oblique (non–axis-aligned) direction which is the
    //   main difference to the other tests.
    // - Linear velocities are almost aligned with the line of centers, producing only minimal
    //   tangential forces. However these need to be captured correctly otherwise the test should
    //   fail.
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

    const std::map<ParticleTestData::particle_id_type, dealii::Tensor<1, dim, double>>
      expected_forces_map = {{0, dealii::Tensor<1, dim, double>({-2672.2552, -4421.6744, 0.0})},
                             {1, dealii::Tensor<1, dim, double>({2672.2552, 4421.6744, 0.0})}};

    for (const auto &particle : this->obstacle_field->get_particle_handler())
      {
        const ParticleTestData::particle_id_type particle_id =
          static_cast<ParticleTestData::particle_id_type>(
            ObstacleType::get_property(particle, ObstacleType::Properties::particle_id));
        const dealii::Tensor<1, dim, double> computed_force = ObstacleType::get_force(particle);
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_NEAR(computed_force[d], expected_force[d], 1e-4)
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }
} // namespace
