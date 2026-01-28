
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/particles/cohesive_forces.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>

#include <numbers>

namespace MeltPoolDG
{
  template <typename number>
  void
  SphericalParticleCohesiveForceData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("cohesive forces");
    {
      prm.add_parameter(
        "hamaker constant",
        hamaker_constant,
        "Hamaker constant used in the van der Waals force calculation for all particles.");
      prm.add_parameter("surface energy",
                        surface_energy,
                        "Surface energy used in the pull-off force calculation for all particles.");
      prm.add_parameter(
        "cut off relative decline van der waals",
        cut_off_relative_decline_van_der_waals,
        "Cut-off relative decline for the van der Waals force calculation at which the force is set to zero.");
    }
    prm.leave_subsection();
  }

  template <int dim, typename number, typename ObstacleType>
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::SphericalParticleCohesiveForce(
    const SphericalParticleCohesiveForceData<number> &cohesive_force_data)
    : cohesive_force_data(cohesive_force_data)
  {}

  template <int dim, typename number, typename ObstacleType>
  void
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::add_load_to_obstacles(
    ObstacleField<dim, number, ObstacleType> &obstacle_field) const
  {
    for (DEMParticleAccessor<dim, number> &particle : obstacle_field.locally_owned_particle_range())
      {
        for (DEMParticleAccessor<dim, number> &other : obstacle_field.global_particle_range())
          {
            if (particle.id() == other.id())
              continue;

            CohesiveContactConfiguration contact_configuration(particle,
                                                               other,
                                                               cohesive_force_data);

            if (contact_configuration.normal_distance < contact_configuration.cut_off_distance)
              {
                dealii::Tensor<1, dim, number> cohesive_force;
                if (contact_configuration.normal_distance <=
                    contact_configuration.pull_off_force_limit_distance)
                  {
                    cohesive_force = contact_configuration.pull_off_force_magnitude *
                                     contact_configuration.normal_vector;
                  }
                else
                  {
                    cohesive_force = cohesive_force_data.hamaker_constant *
                                     contact_configuration.effective_radius /
                                     (6. * contact_configuration.normal_distance *
                                      contact_configuration.normal_distance) *
                                     contact_configuration.normal_vector;
                  }
                particle.add_force(cohesive_force);
              }
          }
      }
  }

  template <int dim, typename number, typename ObstacleType>
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::CohesiveContactConfiguration::
    CohesiveContactConfiguration(
      const DEMParticleAccessor<dim, number>           &self,
      const DEMParticleAccessor<dim, number>           &other,
      const SphericalParticleCohesiveForceData<number> &cohesive_force_data)
  {
    const number distance = self.get_location().distance(other.get_location());
    normal_vector         = (other.get_location() - self.get_location()) / distance;

    normal_distance  = distance - (self.get_property(ObstacleType::Properties::radius) +
                                  other.get_property(ObstacleType::Properties::radius));
    effective_radius = (self.get_property(ObstacleType::Properties::radius) *
                        other.get_property(ObstacleType::Properties::radius)) /
                       (self.get_property(ObstacleType::Properties::radius) +
                        other.get_property(ObstacleType::Properties::radius));

    pull_off_force_magnitude =
      4. * std::numbers::pi_v<number> * cohesive_force_data.surface_energy * effective_radius;

    pull_off_force_limit_distance =
      std::sqrt(cohesive_force_data.hamaker_constant * effective_radius /
                (6. * std::abs(pull_off_force_magnitude)));

    cut_off_distance = pull_off_force_limit_distance /
                       std::sqrt(cohesive_force_data.cut_off_relative_decline_van_der_waals);
  }
} // namespace MeltPoolDG

template struct MeltPoolDG::SphericalParticleCohesiveForceData<double>;

template class MeltPoolDG::
  SphericalParticleCohesiveForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::
  SphericalParticleCohesiveForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::
  SphericalParticleCohesiveForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;
