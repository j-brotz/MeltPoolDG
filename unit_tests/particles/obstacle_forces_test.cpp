
#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <memory>

namespace
{
  constexpr int dim = 3;

  class ObstacleForceFixture : public ::testing::Test
  {
  protected:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

    ObstacleForceFixture()
      : mapping(1)
      , timer(std::cout, dealii::TimerOutput::never, dealii::TimerOutput::wall_times)
    {}

    void
    SetUp() override
    {
      dealii::GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(4);

      // Use three particles with different properties
      const auto create_particle_properties = [](const double mass,
                                                 const int    particle_id) -> std::vector<double> {
        std::vector<double> properties(ObstacleType::n_obstacle_properties, 0);
        properties[ObstacleType::Properties::mass]        = mass;
        properties[ObstacleType::Properties::particle_id] = particle_id;
        return properties;
      };

      // dummy properties with different masses
      std::vector<std::vector<double>> particle_properties = {
        create_particle_properties(2.2e-12, 0), // (approx.) radius 5e-6m, density 4300 kg/m^3
        create_particle_properties(2.8e-10, 1), // (approx.) radius 25e-6m, density 4300 kg/m^3
        create_particle_properties(2.2e-9, 2)   // (approx.) radius 50e-6m, density 4300 kg/m^3
      };

      // dummy locations
      std::vector<dealii::Point<dim>> particle_locations = {dealii::Point<dim>(0.25, 0.25, 0.5),
                                                            dealii::Point<dim>(0.5, 0.5, 0.25),
                                                            dealii::Point<dim>(0.75, 0.75, 0.6)};

      ASSERT_EQ(particle_locations.size(), particle_properties.size());

      obstacle_field = std::make_unique<MeltPoolDG::ObstacleField<dim, double, ObstacleType>>(
        obstacle_data, triangulation, mapping, particle_locations, particle_properties, timer);
    }

    dealii::Triangulation<dim>                                            triangulation;
    dealii::MappingQ<dim>                                                 mapping;
    MeltPoolDG::ObstacleData<double>                                      obstacle_data;
    std::unique_ptr<MeltPoolDG::ObstacleField<dim, double, ObstacleType>> obstacle_field;
    dealii::TimerOutput                                                   timer;
  };

  TEST_F(ObstacleForceFixture, GravityForce)
  {
    using particle_id_type = int;

    constexpr double gravitational_constant = 10;

    const MeltPoolDG::ObstacleGravitationalForce<dim, double, ObstacleType> gravity_force(
      gravitational_constant);
    gravity_force.add_load_to_obstacles(*this->obstacle_field);

    const std::map<particle_id_type, dealii::Tensor<1, dim, double>> expected_forces_map = {
      {0, dealii::Tensor<1, dim, double>({0.0, 0.0, -2.2e-11})},
      {1, dealii::Tensor<1, dim, double>({0.0, 0.0, -2.8e-9})},
      {2, dealii::Tensor<1, dim, double>({0.0, 0.0, -2.2e-8})}};

    for (const auto &particle : this->obstacle_field->get_particle_handler())
      {
        const particle_id_type particle_id = static_cast<particle_id_type>(
          ObstacleType::get_property(particle, ObstacleType::Properties::particle_id));
        const dealii::Tensor<1, dim, double> computed_force = ObstacleType::get_force(particle);
        const dealii::Tensor<1, dim, double> expected_force = expected_forces_map.at(particle_id);

        for (int d = 0; d < dim; ++d)
          {
            EXPECT_DOUBLE_EQ(computed_force[d], expected_force[d])
              << "Particle id: " << particle_id << ", force component: " << d;
          }
      }
  }
} // namespace
