#pragma once

#include <deal.II/base/timer.h>

#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/time_integration/solution_history.hpp>

namespace MeltPoolDG
{
  /**
   * @brief Advances particle states by one time step using the symplectic Euler scheme.
   *
   * Updates translational and angular velocities and positions in place according to
   * the symplectic (semi-implicit) Euler integration method:
   * \f[
   * \mathbf{v}^{(n+1)} &= \mathbf{v}^{(n)} + \Delta t \mathbf{a}^{(n)},\\
   * \mathbf{x}^{(n+1)} &= \mathbf{x}^{(n)} + \Delta t \mathbf{v}^{(n+1)},\\
   * \boldsymbol{\omega}^{(n+1)} &= \boldsymbol{\omega}^{(n)} + \Delta t
   * \boldsymbol{\alpha}^{(n)}.
   * \f]
   *
   * where:
   * - \f$\mathbf{x}\f$ is the particle position (`location`)
   * - \f$\mathbf{v}\f$ is the translational velocity (`translational_velocity`)
   * - \f$\mathbf{a}\f$ is the translational acceleration (`translational_acceleration`)
   * - \f$\boldsymbol{\omega}\f$ is the angular velocity (`angular_velocity`)
   * - \f$\boldsymbol{\alpha}\f$ is the angular acceleration (`angular_acceleration`)
   * - \f$\Delta t\f$ is the time step size (`time_step`)
   * - superscripts \f$(n)\f$, \f$(n+1)\f$ denote the current and next points in time
   *
   * At the end of the update, all particles in the provided particle range, i.e. location,
   * translational_velocity, and so on have the value at the new time step. This includes the ghost
   * values which are updated internally.
   *
   * @param timer Timer for performance measurement.
   * @param current_time Current simulation time.
   * @param time_step Time step size.
   * @param particle_range Range of particles to update.
   */
  template <int dim, typename number, typename ObstacleType>
  void
  symplectic_euler_advance_time_step(
    dealii::TimerOutput                                 &timer,
    const number                                         current_time,
    const number                                         time_step,
    std::ranges::subrange<ParticleIterator<dim, number>> particle_range)
  {
    dealii::TimerOutput::Scope t(timer, "DEM time integrator");
    for (DEMParticleAccessor<dim, number> &particle : particle_range)
      {
        for (unsigned d = 0; d < dim; ++d)
          {
            particle.linear_velocity(d) += time_step * particle.linear_acceleration(d);
            particle.get_location()[d] += time_step * particle.linear_velocity(d);
            particle.linear_acceleration(d) =
              particle.force(d) / particle.get_property(ObstacleType::Properties::mass);
          }

        for (unsigned d = 0; d < axial_dim<dim>; ++d)
          {
            particle.angular_velocity(d) += time_step * particle.angular_acceleration(d);
            particle.angular_acceleration(d) =
              particle.torque(d) /
              particle.get_property(ObstacleType::Properties::moment_of_inertia);
          }
      }
  }
} // namespace MeltPoolDG
