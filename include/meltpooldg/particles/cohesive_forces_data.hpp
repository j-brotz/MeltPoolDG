#pragma once

#include <deal.II/base/parameter_handler.h>

namespace MeltPoolDG
{
  template <typename number>
  struct SphericalParticleCohesiveForceData
  {
    /// Hamaker constant used in the van der Waals force calculation for all particles.
    number hamaker_constant{};

    /// Surface energy used in the pull-off force calculation for all particles.
    number surface_energy{};

    /// Cut-off relative decline for the van der Waals force calculation at which the force is set
    /// to zero.
    number cut_off_relative_decline_van_der_waals{};

    /**
     * Add the relevant parameters to a parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("cohesive forces");
      {
        prm.add_parameter(
          "hamaker constant",
          hamaker_constant,
          "Hamaker constant used in the van der Waals force calculation for all particles.");
        prm.add_parameter(
          "surface energy",
          surface_energy,
          "Surface energy used in the pull-off force calculation for all particles.");
        prm.add_parameter(
          "cut off relative decline van der waals",
          cut_off_relative_decline_van_der_waals,
          "Cut-off relative decline for the van der Waals force calculation at which the force is set to zero.");
      }
      prm.leave_subsection();
    }
  };
} // namespace MeltPoolDG
