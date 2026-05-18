#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  template <typename number>
  struct SphericalParticleContactData
  {
    /// Coefficient of restitution for damping in particle collisions.
    number restitution_coefficient{};

    /// Sliding friction coefficient for Coulomb friction model.
    number sliding_friction_coefficient{};

    /// Coefficient of rolling resistance used to compute rolling resistance torques.
    number rolling_resistance_coefficient{};

    struct MaterialData
    {
      /// Young's modulus of the particle material.
      number youngs_modulus{};

      /// Poisson's ratio of the particle material.
      number poisson_ratio{};
    };

    /// Material data for the particles. Those are assumed to be identical for all particles.
    MaterialData particle;

    /**
     * Add the relevant parameters to a parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("spherical particle contact force");
      {
        prm.add_parameter("restitution coefficient", restitution_coefficient);
        prm.add_parameter("sliding friction coefficient", sliding_friction_coefficient);
        prm.add_parameter("rolling resistance coefficient", rolling_resistance_coefficient);
        prm.enter_subsection("particle-particle contact");
        {
          prm.add_parameter("youngs modulus", particle.youngs_modulus);
          prm.add_parameter("poisson ratio", particle.poisson_ratio);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
