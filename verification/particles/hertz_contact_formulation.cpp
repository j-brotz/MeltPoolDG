
#include <gtest/gtest.h>

#include <deal.II/base/function_signed_distance.h>

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

#include <memory>
#include <numbers>

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

  template <int dim>
  void
  time_loop(MeltPoolDG::TimeIntegration::TimeIterator<double> &time_iterator,
            MeltPoolDG::ObstacleField<dim, double, MeltPoolDG::SphericalParticle<dim, double>>
                                        &obstacle_field,
            const std::function<void()> &post_processor = std::function<void()>())
  {
    while (!time_iterator.is_finished())
      {
        obstacle_field.advance_time(time_iterator.get_current_time(),
                                    time_iterator.get_current_time_increment());
        time_iterator.compute_next_time_increment();
        if (post_processor)
          post_processor();
      }
  }

  class ContactForceFixture : public ::testing::Test
  {
  protected:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;
    struct Data
    {
      Data(const MeltPoolDG::ObstacleData<double>                      &obstacle_data,
           const MeltPoolDG::SphericalParticleContactData<double>      &contact_data,
           const MeltPoolDG::TimeIntegration::TimeSteppingData<double> &time_integration_data,
           std::vector<std::unique_ptr<dealii::Function<dim>>>        &&walls = {})
        : mapping(1)
        , obstacle_data(obstacle_data)
        , contact_data(contact_data)
        , time_iterator(time_integration_data)
      {
        dealii::GridGenerator::hyper_cube(triangulation);
        triangulation.refine_global(2);

        MeltPoolDG::SphericalParticleContactForce<dim, double, ObstacleType> contact_force(
          contact_data, time_iterator);

        for (auto &wall : walls)
          contact_force.attach_wall(std::move(wall));

        std::vector<std::vector<double>> particle_properties = {};
        std::vector<dealii::Point<dim>>  particle_locations  = {};

        obstacle_field = std::make_unique<MeltPoolDG::ObstacleField<dim, double, ObstacleType>>(
          obstacle_data, triangulation, mapping, particle_locations, particle_properties);
        obstacle_field->add_load_type(std::move(contact_force));
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
        properties[ObstacleType::Properties::moment_of_inertia] =
          dim == 3 ? (2.0 / 5.0) * properties[ObstacleType::Properties::mass] *
                       particle_data.radius * particle_data.radius :
                     (1.0 / 2.0) * properties[ObstacleType::Properties::mass] *
                       particle_data.radius * particle_data.radius;

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
    };

    std::unique_ptr<Data> data;
  };

  TEST_F(ContactForceFixture, ParticleParticleElasticNormalImpact)
  {
    data = std::make_unique<Data>(MeltPoolDG::ObstacleData<double>{},
                                  MeltPoolDG::SphericalParticleContactData<double>{
                                    .restitution_coefficient      = 1.0,
                                    .sliding_friction_coefficient = 0.0,
                                    .particle = {.youngs_modulus = 4.8e10, .poisson_ratio = 0.2}},
                                  MeltPoolDG::TimeIntegration::TimeSteppingData<double>{
                                    .start_time = 0.0, .end_time = 2e-4, .time_step_size = 1e-8});

    constexpr double particle_radius  = 0.01;
    constexpr double particle_density = 2800;

    data->insert_particle({.id               = 0,
                           .radius           = particle_radius,
                           .density          = particle_density,
                           .angular_velocity = dealii::Tensor<1, dim, double>({0., 0., 0.}),
                           .linear_velocity  = dealii::Tensor<1, dim, double>({10.0, 0, 0}),
                           .location         = dealii::Point<dim>(0.489, 0.5, 0.5)});
    data->insert_particle({.id               = 1,
                           .radius           = particle_radius,
                           .density          = particle_density,
                           .angular_velocity = dealii::Tensor<1, dim, double>({0., 0., 0.}),
                           .linear_velocity  = dealii::Tensor<1, dim, double>({-10.0, 0, 0}),
                           .location         = dealii::Point<dim>(0.511, 0.5, 0.5)});

    double max_normal_contact_force        = 0.0;
    double max_normal_contact_displacement = 0.0;
    double contact_duration                = 0.0;
    auto   post_processor                  = [&]() {
      max_normal_contact_force =
        std::max(max_normal_contact_force,
                 std::abs(data->obstacle_field->begin()->get_force().norm()));

      auto particle_iterator = data->obstacle_field->begin();

      auto loc_particle1 = particle_iterator->get_location();
      particle_iterator++;
      auto loc_particle2 = particle_iterator->get_location();
      max_normal_contact_displacement =
        std::max(max_normal_contact_displacement,
                 2 * particle_radius - std::abs(loc_particle2[0] - loc_particle1[0]));

      if (2 * particle_radius - std::abs(loc_particle2[0] - loc_particle1[0]) > 0)
        contact_duration += data->time_iterator.get_current_time_increment();
    };

    time_loop(data->time_iterator, *data->obstacle_field, post_processor);

    double helper_factor =
      std::pow(5 * std::numbers::sqrt2_v<double> * particle_density * std::numbers::pi_v<double> *
                 (1 - data->contact_data.particle.poisson_ratio *
                        data->contact_data.particle.poisson_ratio) /
                 (4. * data->contact_data.particle.youngs_modulus),
               0.4);
    double expected_contact_duration =
      2.943 * helper_factor * particle_radius / std::pow(20.0, 0.2);

    double expected_max_normal_contact_displacement =
      helper_factor * particle_radius * std::pow(20.0, 0.8);

    double expected_max_normal_contact_force =
      std::sqrt((2. * particle_radius * data->contact_data.particle.youngs_modulus *
                 data->contact_data.particle.youngs_modulus) /
                (9. * std::pow(1 - data->contact_data.particle.poisson_ratio *
                                     data->contact_data.particle.poisson_ratio,
                               2))) *
      std::pow(expected_max_normal_contact_displacement, 1.5);

    constexpr double relative_tolerance = 1e-3;
    EXPECT_NEAR(max_normal_contact_force,
                expected_max_normal_contact_force,
                relative_tolerance * expected_max_normal_contact_force);
    EXPECT_NEAR(max_normal_contact_displacement,
                expected_max_normal_contact_displacement,
                relative_tolerance * expected_max_normal_contact_displacement);
    EXPECT_NEAR(contact_duration,
                expected_contact_duration,
                relative_tolerance * expected_contact_duration);
  }

  TEST_F(ContactForceFixture, ParticleWallElasticNormalImpact)
  {
    std::vector<std::unique_ptr<dealii::Function<dim>>> walls;
    walls.emplace_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
      dealii::Point<dim>(0, 0, 0), dealii::Tensor<1, dim>({0, 0, 1})));

    data = std::make_unique<Data>(MeltPoolDG::ObstacleData<double>{},
                                  MeltPoolDG::SphericalParticleContactData<double>{
                                    .restitution_coefficient      = 1.0,
                                    .sliding_friction_coefficient = 0.0,
                                    .particle = {.youngs_modulus = 7e10, .poisson_ratio = 0.3}},
                                  MeltPoolDG::TimeIntegration::TimeSteppingData<double>{
                                    .start_time = 0.0, .end_time = 2e-3, .time_step_size = 1e-8},
                                  std::move(walls));

    constexpr double particle_radius            = 0.1;
    constexpr double particle_density           = 2699;
    constexpr double particle_relative_velocity = -0.2;

    data->insert_particle(
      {.id               = 0,
       .radius           = particle_radius,
       .density          = particle_density,
       .angular_velocity = dealii::Tensor<1, dim, double>({0., 0., 0.}),
       .linear_velocity  = dealii::Tensor<1, dim, double>({0, 0, particle_relative_velocity}),
       .location         = dealii::Point<dim>(0.5, 0.5, 0.10001)});

    double max_normal_contact_force        = 0.0;
    double max_normal_contact_displacement = 0.0;
    double contact_duration                = 0.0;
    auto   post_processor                  = [&]() {
      max_normal_contact_force =
        std::max(max_normal_contact_force,
                 std::abs(data->obstacle_field->begin()->get_force().norm()));

      auto particle_iterator = data->obstacle_field->begin();

      auto loc_particle = particle_iterator->get_location();
      particle_iterator++;
      max_normal_contact_displacement =
        std::max(max_normal_contact_displacement, particle_radius - std::abs(loc_particle[2]));

      if (particle_radius - std::abs(loc_particle[2]) > 0)
        contact_duration += data->time_iterator.get_current_time_increment();
    };

    time_loop(data->time_iterator, *data->obstacle_field, post_processor);

    const double helper_factor =
      std::pow(5 * std::numbers::sqrt2_v<double> * particle_density * std::numbers::pi_v<double> *
                 (1 - data->contact_data.particle.poisson_ratio *
                        data->contact_data.particle.poisson_ratio) /
                 (4. * data->contact_data.particle.youngs_modulus),
               0.4);
    const double expected_contact_duration =
      2.943 * helper_factor * particle_radius /
      std::pow(2 * std::abs(particle_relative_velocity), 0.2);

    double expected_max_normal_contact_displacement =
    helper_factor *particle_radius *std::pow(2 * std::abs(particle_relative_velocity), 0.8);

    const double expected_max_normal_contact_force =
      std::sqrt((2. * particle_radius * data->contact_data.particle.youngs_modulus *
                 data->contact_data.particle.youngs_modulus) /
                (9. * std::pow(1 - data->contact_data.particle.poisson_ratio *
                                     data->contact_data.particle.poisson_ratio,
                               2))) *
      std::pow(expected_max_normal_contact_displacement, 1.5);
      expected_max_normal_contact_displacement *= 0.5;


    constexpr double relative_tolerance = 1e-3;
    EXPECT_NEAR(max_normal_contact_force,
                expected_max_normal_contact_force,
                relative_tolerance * expected_max_normal_contact_force);
    EXPECT_NEAR(max_normal_contact_displacement,
                expected_max_normal_contact_displacement,
                relative_tolerance * expected_max_normal_contact_displacement);
    EXPECT_NEAR(contact_duration,
                expected_contact_duration,
                relative_tolerance * expected_contact_duration);
  }

} // namespace