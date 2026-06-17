#include <gtest/gtest.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/particles/dem_time_integrators.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>

#include <deque>
#include <string>
#include <vector>

namespace
{
  constexpr int dim  = 3;
  using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

  /**
   * Compares a particle's current kinematic state against expected values.
   *
   * @param particle Particle whose current state is checked.
   * @param location Expected location.
   * @param velocity Expected linear velocity.
   * @param acceleration Expected linear acceleration.
   * @param angular_velocity Expected angular velocity.
   * @param angular_acceleration Expected angular acceleration.
   */
  void
  test_particle_state(MeltPoolDG::DEMParticleAccessor<dim, double> &particle,
                      const dealii::Point<dim>                     &location,
                      const dealii::Tensor<1, dim, double>         &velocity,
                      const dealii::Tensor<1, dim, double>         &acceleration,
                      const dealii::Tensor<1, dim, double>         &angular_velocity,
                      const dealii::Tensor<1, dim, double>         &angular_acceleration)
  {
    for (int d = 0; d < dim; ++d)
      {
        SCOPED_TRACE("Dimension " + std::to_string(d));
        EXPECT_DOUBLE_EQ(particle.get_location()[d], location[d]);
        EXPECT_DOUBLE_EQ(particle.linear_velocity(d), velocity[d]);
        EXPECT_DOUBLE_EQ(particle.linear_acceleration(d), acceleration[d]);
      }

    for (int d = 0; d < MeltPoolDG::axial_dim<dim>; ++d)
      {
        SCOPED_TRACE("Axial Dimension " + std::to_string(d));
        EXPECT_DOUBLE_EQ(particle.angular_velocity(d), angular_velocity[d]);
        EXPECT_DOUBLE_EQ(particle.angular_acceleration(d), angular_acceleration[d]);
      }
  }

  class SymplecticEulerTest : public ::testing::Test
  {
  protected:
    /// Accessors for all particles added via add_particle(), in insertion order.
    std::vector<MeltPoolDG::DEMParticleAccessor<dim, double>> particles;

    /**
     * Creates a particle with the given kinematic state, stores the state internally and adds an
     * accessor for the particle to the particles vector member variable.
     *
     * @param location Location of the particle.
     * @param velocity Linear velocity of the particle.
     * @param acceleration Linear acceleration of the particle.
     * @param angular_velocity Angular velocity of the particle.
     * @param angular_acceleration Angular acceleration of the particle.
     * @param force Net force acting on the particle.
     * @param torque Net torque acting on the particle.
     * @param mass Particle mass.
     */
    void
    add_particle(dealii::Point<dim>             location,
                 dealii::Tensor<1, dim, double> velocity,
                 dealii::Tensor<1, dim, double> acceleration,
                 dealii::Tensor<1, dim, double> angular_velocity,
                 dealii::Tensor<1, dim, double> angular_acceleration,
                 dealii::Tensor<1, dim, double> force,
                 dealii::Tensor<1, dim, double> torque,
                 double                         mass,
                 double                         moment_of_inertia)
    {
      locations.push_back(location);
      storage.emplace_back(ObstacleType::n_obstacle_properties, 0.0);
      MeltPoolDG::DEMParticleAccessor<dim, double> particle(locations.back(), storage.back());
      for (int d = 0; d < dim; ++d)
        {
          particle.linear_velocity(d)      = velocity[d];
          particle.linear_acceleration(d)  = acceleration[d];
          particle.angular_velocity(d)     = angular_velocity[d];
          particle.angular_acceleration(d) = angular_acceleration[d];
          particle.force(d)                = force[d];
          particle.torque(d)               = torque[d];
        }
      particle.get_property(ObstacleType::Properties::mass)              = mass;
      particle.get_property(ObstacleType::Properties::moment_of_inertia) = moment_of_inertia;
      particles.push_back(particle);
    }

  private:
    /// Storage of the particle properties
    std::deque<std::vector<double>> storage;

    /// Storage of the particle locations
    std::deque<dealii::Point<dim>> locations;
  };
} // namespace

/**
 * Tests the symplectic Euler time integration for a single particle in 3D advancing one time step.
 */
TEST_F(SymplecticEulerTest, 3DSingleParticle)
{
  // Initialize a single particle with specific properties
  dealii::Point<dim> initial_location(1.0, 2.0, 3.0);

  dealii::Tensor<1, dim, double> initial_velocity({0.5, 2.0, -0.5});
  dealii::Tensor<1, dim, double> initial_acceleration({0.1, -0.2, 0.05});
  dealii::Tensor<1, dim, double> initial_angular_velocity({0.01, 0.02, 0.03});
  dealii::Tensor<1, dim, double> initial_angular_acceleration({0.001, -0.002, 0.003});
  dealii::Tensor<1, dim, double> force({1.5, 2.0, -0.5});
  dealii::Tensor<1, dim, double> torque({0.1, 0.2, -0.1});
  double                         mass              = 2.0;
  double                         moment_of_inertia = 0.5;

  this->add_particle(initial_location,
                     initial_velocity,
                     initial_acceleration,
                     initial_angular_velocity,
                     initial_angular_acceleration,
                     force,
                     torque,
                     mass,
                     moment_of_inertia);

  // Advance one time step using symplectic Euler method
  double time_step = 0.1;
  MeltPoolDG::symplectic_euler_advance_time_step<dim, double, ObstacleType>(time_step,
                                                                            this->particles);

  // Test new state of the particle after one time step
  dealii::Point<dim>             expected_location(1.051, 2.198, 2.9505);
  dealii::Tensor<1, dim, double> expected_velocity({0.51, 1.98, -0.495});
  dealii::Tensor<1, dim, double> expected_acceleration({0.75, 1.0, -0.25});
  dealii::Tensor<1, dim, double> expected_angular_velocity({0.0101, 0.0198, 0.0303});
  dealii::Tensor<1, dim, double> expected_angular_acceleration({0.2, 0.4, -0.2});
  test_particle_state(this->particles[0],
                      expected_location,
                      expected_velocity,
                      expected_acceleration,
                      expected_angular_velocity,
                      expected_angular_acceleration);
}

/**
 * Tests the symplectic Euler time integration for a single particle in 3D advancing a time step of
 * zero. This test should check that the state of the particle remains unchanged but the new
 * accelerations are updated according to the provided forces and torques.
 */
TEST_F(SymplecticEulerTest, 3DZeroTimeStep)
{
  // Initialize a single particle with specific properties
  dealii::Point<dim> initial_location(1.0, 2.0, 3.0);

  dealii::Tensor<1, dim, double> initial_velocity({0.5, 2.0, -0.5});
  dealii::Tensor<1, dim, double> initial_acceleration({0.1, -0.2, 0.05});
  dealii::Tensor<1, dim, double> initial_angular_velocity({0.01, 0.02, 0.03});
  dealii::Tensor<1, dim, double> initial_angular_acceleration({0.001, -0.002, 0.003});
  dealii::Tensor<1, dim, double> force({1.5, 2.0, -0.5});
  dealii::Tensor<1, dim, double> torque({0.1, 0.2, -0.1});
  double                         mass              = 2.0;
  double                         moment_of_inertia = 0.5;

  this->add_particle(initial_location,
                     initial_velocity,
                     initial_acceleration,
                     initial_angular_velocity,
                     initial_angular_acceleration,
                     force,
                     torque,
                     mass,
                     moment_of_inertia);

  // Advance one time step using symplectic Euler method with zero time step
  double time_step = 0.0;
  MeltPoolDG::symplectic_euler_advance_time_step<dim, double, ObstacleType>(time_step,
                                                                            this->particles);

  // Check that the state of the particle remains unchanged but the new linear and angular
  // accelerations
  dealii::Point<dim>             expected_location(1.0, 2.0, 3.0);
  dealii::Tensor<1, dim, double> expected_velocity({0.5, 2.0, -0.5});
  dealii::Tensor<1, dim, double> expected_acceleration({0.75, 1.0, -0.25});
  dealii::Tensor<1, dim, double> expected_angular_velocity({0.01, 0.02, 0.03});
  dealii::Tensor<1, dim, double> expected_angular_acceleration({0.2, 0.4, -0.2});
  test_particle_state(this->particles[0],
                      expected_location,
                      expected_velocity,
                      expected_acceleration,
                      expected_angular_velocity,
                      expected_angular_acceleration);
}

/**
 * Tests the symplectic Euler time integration for multiple particles in 3D. The test provided here
 * initializes two particles with specific properties, advances them one time step using the
 * symplectic Euler method, and checks that their new states match the expected values after the
 * time step
 */
TEST_F(SymplecticEulerTest, 3DMultipleParticles)
{
  // Initialize two particles with specific properties
  dealii::Point<dim>             initial_location_1(1.0, 2.0, 3.0);
  dealii::Tensor<1, dim, double> initial_velocity_1({0.5, 2.0, -0.5});
  dealii::Tensor<1, dim, double> initial_acceleration_1({0.1, -0.2, 0.05});
  dealii::Tensor<1, dim, double> initial_angular_velocity_1({0.01, 0.02, 0.03});
  dealii::Tensor<1, dim, double> initial_angular_acceleration_1({0.001, -0.002, 0.003});
  dealii::Tensor<1, dim, double> force_1({1.5, 2.0, -0.5});
  dealii::Tensor<1, dim, double> torque_1({0.1, 0.2, -0.1});
  double                         mass_1              = 2.0;
  double                         moment_of_inertia_1 = 0.5;

  this->add_particle(initial_location_1,
                     initial_velocity_1,
                     initial_acceleration_1,
                     initial_angular_velocity_1,
                     initial_angular_acceleration_1,
                     force_1,
                     torque_1,
                     mass_1,
                     moment_of_inertia_1);

  dealii::Point<dim>             initial_location_2(3.0, 5.0, 2.0);
  dealii::Tensor<1, dim, double> initial_velocity_2({1.5, 6.0, -1.5});
  dealii::Tensor<1, dim, double> initial_acceleration_2({0.7, -0.3, 0.08});
  dealii::Tensor<1, dim, double> initial_angular_velocity_2({0.03, 0.06, 0.09});
  dealii::Tensor<1, dim, double> initial_angular_acceleration_2({0.004, -0.001, 0.002});
  dealii::Tensor<1, dim, double> force_2({1.8, 2.3, -0.7});
  dealii::Tensor<1, dim, double> torque_2({0.3, 0.4, -0.2});
  double                         mass_2              = 2.0;
  double                         moment_of_inertia_2 = 0.5;

  this->add_particle(initial_location_2,
                     initial_velocity_2,
                     initial_acceleration_2,
                     initial_angular_velocity_2,
                     initial_angular_acceleration_2,
                     force_2,
                     torque_2,
                     mass_2,
                     moment_of_inertia_2);

  // Advance one time step using symplectic Euler method
  double time_step = 0.1;
  MeltPoolDG::symplectic_euler_advance_time_step<dim, double, ObstacleType>(time_step,
                                                                            this->particles);

  // Test new state of the particles after one time step
  {
    SCOPED_TRACE("Particle 1");
    dealii::Point<dim>             expected_location_1(1.051, 2.198, 2.9505);
    dealii::Tensor<1, dim, double> expected_velocity_1({0.51, 1.98, -0.495});
    dealii::Tensor<1, dim, double> expected_acceleration_1({0.75, 1.0, -0.25});
    dealii::Tensor<1, dim, double> expected_angular_velocity_1({0.0101, 0.0198, 0.0303});
    dealii::Tensor<1, dim, double> expected_angular_acceleration_1({0.2, 0.4, -0.2});
    test_particle_state(this->particles[0],
                        expected_location_1,
                        expected_velocity_1,
                        expected_acceleration_1,
                        expected_angular_velocity_1,
                        expected_angular_acceleration_1);
  }

  {
    SCOPED_TRACE("Particle 2");
    dealii::Point<dim>             expected_location_2(3.157, 5.597, 1.8508);
    dealii::Tensor<1, dim, double> expected_velocity_2({1.57, 5.97, -1.492});
    dealii::Tensor<1, dim, double> expected_acceleration_2({0.9, 1.15, -0.35});
    dealii::Tensor<1, dim, double> expected_angular_velocity_2({0.0304, 0.0599, 0.0902});
    dealii::Tensor<1, dim, double> expected_angular_acceleration_2({0.6, 0.8, -0.4});
    test_particle_state(this->particles[1],
                        expected_location_2,
                        expected_velocity_2,
                        expected_acceleration_2,
                        expected_angular_velocity_2,
                        expected_angular_acceleration_2);
  }
}
