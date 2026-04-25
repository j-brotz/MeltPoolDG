#pragma once

#include <deal.II/base/timer.h>

#include <meltpooldg/time_integration/solution_history.hpp>

#include <functional>

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
   * At the end of the update, all provided vector views, i.e. location, translational_velocity, and
   * so on have the value at the new time step. This includes the ghost values which are updated
   * internally.
   *
   * @param current_time Current simulation time.
   * @param time_step Time step size.
   * @param location Particle position history.
   * @param translational_velocity Translational velocity history.
   * @param translational_acceleration Translational acceleration history.
   * @param angular_velocity Angular velocity history.
   * @param angular_acceleration Angular acceleration history.
   * @param compute_translational_acceleration Function that computes translational acceleration
   * given the current time and writes it into the provided vector.
   * @param compute_angular_acceleration Function that computes angular acceleration
   * given the current time and writes it into the provided vector.
   * @param update_ghost_values Optional function to update ghost values. If this function is passed
   * the implementation will not call any of the vectors update ghost values functionality but only
   * the provided one.
   *
   * @note The updates are performed directly on the current solution views.
   */
  template <int dim, typename number, typename BlockVectorType>
  void
  symplectic_euler_advance_time_step(
    dealii::TimerOutput                               &timer,
    const number                                       current_time,
    const number                                       time_step,
    TimeIntegration::SolutionHistory<BlockVectorType> &location,
    TimeIntegration::SolutionHistory<BlockVectorType> &translational_velocity,
    TimeIntegration::SolutionHistory<BlockVectorType> &translational_acceleration,
    TimeIntegration::SolutionHistory<BlockVectorType> &angular_velocity,
    TimeIntegration::SolutionHistory<BlockVectorType> &angular_acceleration,
    const std::function<void(number time, BlockVectorType &dst)>
      &compute_translational_acceleration,
    const std::function<void(number time, BlockVectorType &dst)> &compute_angular_acceleration,
    const std::function<void()> &update_ghost_values = std::function<void()>())
  {
    dealii::TimerOutput::Scope t(timer, "DEM time integrator");
    constexpr unsigned size_angular_velocity = dim - 3 % dim;
    for (unsigned i = 0; i < location.get_current_solution().block(0).locally_owned_size(); ++i)
      {
        for (unsigned j = 0; j < dim; ++j)
          {
            translational_velocity.get_current_solution().block(j).local_element(i) +=
              time_step *
              translational_acceleration.get_current_solution().block(j).local_element(i);
            location.get_current_solution().block(j).local_element(i) +=
              time_step * translational_velocity.get_current_solution().block(j).local_element(i);
          }
        for (unsigned j = 0; j < size_angular_velocity; ++j)
          angular_velocity.get_current_solution().block(j).local_element(i) +=
            time_step * angular_acceleration.get_current_solution().block(j).local_element(i);
      }

    if (update_ghost_values)
      update_ghost_values();
    else
      {
        location.update_ghost_values();
        translational_velocity.update_ghost_values();
        angular_velocity.update_ghost_values();
      }

    compute_translational_acceleration(current_time,
                                       translational_acceleration.get_current_solution());
    compute_angular_acceleration(current_time, angular_acceleration.get_current_solution());

    if (update_ghost_values)
      update_ghost_values();
    else
      {
        translational_acceleration.update_ghost_values();
        angular_acceleration.update_ghost_values();
      }
  }
} // namespace MeltPoolDG
