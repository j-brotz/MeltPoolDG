#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/cohesive_forces.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <map>
#include <memory>
#include <string>

/**
 * This test suite verifies the correct implementation of cohesive forces between particles, which
 * are computed as described in Meier et al., “Modeling and characterization of cohesion in fine
 * metal powders with a focus on additive manufacturing process simulations”,
 * (DOI:10.1016/j.powtec.2018.11.072).
 *
 * To this end, several particle configurations are considered, and the resulting cohesive forces
 * are compared against precomputed reference values. See the individual tests for further details.
 */
namespace
{
  constexpr int dim = 3;

  struct ParticleTestData
  {
    using particle_id_type = int;

    particle_id_type           id{};
    double                     radius{};
    dealii::Point<dim, double> location{};
  };

  struct TestData
  {
    std::string                    test_name;
    dealii::Tensor<1, dim, double> expected_force;
    double                         normal_distance;
  };

  class CohesiveForceTest : public ::testing::TestWithParam<TestData>
  {
  protected:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

    CohesiveForceTest()
      : mapping(1)
      , cohesive_force_data({.hamaker_constant                       = 40e-20,
                             .surface_energy                         = 1e-4,
                             .cut_off_relative_decline_van_der_waals = 1e-2})
      , cohesive_force(cohesive_force_data)
      , timer(std::cout, dealii::TimerOutput::never, dealii::TimerOutput::wall_times)
    {}

    void
    SetUp() override
    {
      dealii::GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(4);

      std::vector<std::vector<double>> particle_properties = {};
      std::vector<dealii::Point<dim>>  particle_locations  = {};

      constexpr double particle_influence_radius                       = 0.4;
      obstacle_data.data_structure_data.max_sphere_of_influence_radius = particle_influence_radius;

      obstacle_field = std::make_unique<MeltPoolDG::ObstacleField<dim, double, ObstacleType>>(
        obstacle_data, triangulation, mapping, particle_locations, particle_properties, timer);
    }

    void
    insert_particle(const ParticleTestData &particle_data)
    {
      std::vector<double> properties(ObstacleType::n_obstacle_properties, 0);
      properties[ObstacleType::Properties::radius]      = particle_data.radius;
      properties[ObstacleType::Properties::particle_id] = particle_data.id;

      std::vector<dealii::Point<dim>>  particle_locations  = {particle_data.location};
      std::vector<std::vector<double>> particle_properties = {properties};

      obstacle_field->insert_obstacles({particle_locations}, particle_properties);
    }

    dealii::Triangulation<dim>                                            triangulation;
    dealii::MappingQ<dim>                                                 mapping;
    MeltPoolDG::ObstacleData<double>                                      obstacle_data;
    std::unique_ptr<MeltPoolDG::ObstacleField<dim, double, ObstacleType>> obstacle_field;
    MeltPoolDG::SphericalParticleCohesiveForceData<double>                cohesive_force_data;
    MeltPoolDG::SphericalParticleCohesiveForce<dim, double, ObstacleType> cohesive_force;
    dealii::TimerOutput                                                   timer;
  };

  TEST_P(CohesiveForceTest, ParticleParticleCohesiveForce_DifferentRadii)
  {
    TestData test_data = GetParam();

    insert_particle({.id = 0, .radius = 0.2, .location = dealii::Point<dim>(0.4, 0.5, 0.5)});
    insert_particle({.id       = 1,
                     .radius   = 0.1,
                     .location = dealii::Point<dim>(0.7 + test_data.normal_distance, 0.5, 0.5)});

    cohesive_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<ParticleTestData::particle_id_type, dealii::Tensor<1, dim, double>>
      expected_forces_map = {{0, test_data.expected_force}, {1, {-test_data.expected_force}}};

    for (const MeltPoolDG::DEMParticleAccessor<dim, double> &particle :
         this->obstacle_field->locally_owned_particle_range())
      {
        const ParticleTestData::particle_id_type particle_id =
          static_cast<ParticleTestData::particle_id_type>(particle.id());
        SCOPED_TRACE("Particle id: " + std::to_string(particle_id));

        const dealii::Tensor<1, dim, double> computed_force = particle.get_force();
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            double abs_err = expected_force[d] == 0.0 ? 1e-50 : 1e-7 * std::abs(expected_force[d]);

            SCOPED_TRACE("Component: " + std::to_string(d));
            EXPECT_NEAR(computed_force[d], expected_force[d], abs_err);
          }
      }
  }

  INSTANTIATE_TEST_SUITE_P(
    ParticleParticleCohesiveForce,
    CohesiveForceTest,
    ::testing::Values(
      // Gap is zero, so pull-off force is applied.
      TestData{
        .test_name       = "PullOffForce",
        .expected_force  = dealii::Tensor<1, dim, double>({8.37758041e-5, 0.0, 0.0}),
        .normal_distance = 0.,
      },
      // Gap g_N is chosen such that g_0 < g_N < g*, so van der Waals force is applied.
      TestData{
        .test_name       = "VanDerWaalsForce",
        .expected_force  = dealii::Tensor<1, dim, double>({1.11111111e-5, 0.0, 0.0}),
        .normal_distance = 2e-8,
      },
      TestData{
        // Gap g_N is larger than cut-off distance g*, so no force is applied.
        .test_name       = "ZeroCutOffForce",
        .expected_force  = dealii::Tensor<1, dim, double>({0.0, 0.0, 0.0}),
        .normal_distance = 1e-7,
      }),
    [](const testing::TestParamInfo<CohesiveForceTest::ParamType> &info) {
      return info.param.test_name;
    });
} // namespace
