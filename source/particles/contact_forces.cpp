#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <meltpooldg/particles/contact_forces.hpp>
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>

#include <cmath>
#include <numbers>

namespace MeltPoolDG
{
  template <typename number>
  void
  SphericalParticleContactData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("spherical particle contact force");
    {
      prm.add_parameter("restitution coefficient", restitution_coefficient);
      prm.enter_subsection("particle-particle contact");
      {
        prm.add_parameter("youngs modulus", particle.youngs_modulus);
        prm.add_parameter("poisson ratio", particle.poisson_ratio);
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }


  template <int dim, typename number, typename ObstacleType>
  SphericalParticleContactForce<dim, number, ObstacleType>::SphericalParticleContactForce(
    const SphericalParticleContactData<number> &contact_data)
    : contact_data(contact_data)
    , damping_prefactor(compute_damping_prefactor(contact_data.restitution_coefficient))
  {}


  template <int dim, typename number, typename ObstacleType>
  void
  SphericalParticleContactForce<dim, number, ObstacleType>::add_load_to_obstacles(
    ObstacleField<dim, number, ObstacleType> &obstacle_field) const
  {
    for (auto &particle : obstacle_field.locally_owned_particle_range())
      {
        for (auto &other : obstacle_field.global_particle_range())
          {
            if (particle.id() == other.id())
              continue;

            ContactConfiguration contact_configuration(particle,
                                                       other,
                                                       contact_data.particle.youngs_modulus,
                                                       contact_data.particle.poisson_ratio);

            if (contact_configuration.normal_overlap >= 0)
              {
                dealii::Tensor<1, dim, number> normal_force =
                  normal_contact_force(contact_configuration);
                particle.add_force(normal_force);
              }
          }
      }
  }

  template <int dim, typename number, typename ObstacleType>
  dealii::Tensor<1, dim, number>
  SphericalParticleContactForce<dim, number, ObstacleType>::normal_contact_force(
    const ContactConfiguration &contact_configuration) const
  {
    const number normal_stiffness =
      4. / 3. * contact_configuration.effective_youngs_modulus *
      std::sqrt(contact_configuration.effective_radius * contact_configuration.normal_overlap);

    const number normal_damping =
      damping_prefactor * std::sqrt(1.5 * normal_stiffness * contact_configuration.effective_mass);

    auto normal_force = -normal_stiffness * contact_configuration.normal_overlap *
                          contact_configuration.normal_vector -
                        normal_damping * contact_configuration.relative_velocity.normal_component;
    return normal_force;
  }

  template <int dim, typename number, typename ObstacleType>
  number
  SphericalParticleContactForce<dim, number, ObstacleType>::compute_damping_prefactor(
    const number restitution_coefficient) const
  {
    return -2. * std::sqrt(5. / 6.) * std::log(restitution_coefficient) /
           std::sqrt(dealii::Utilities::fixed_power<2>(std::log(restitution_coefficient)) +
                     dealii::Utilities::fixed_power<2>(std::numbers::pi_v<number>));
  }

  template <int dim, typename number, typename ObstacleType>
  SphericalParticleContactForce<dim, number, ObstacleType>::ContactConfiguration::
    ContactConfiguration(DEMParticleAccessor<dim, number> &self,
                         DEMParticleAccessor<dim, number> &other,
                         const number                      youngs_modulus,
                         const number                      poisson_ratio)
  {
    auto compute_effective_quantity = [](number t1, number t2) -> number {
      return t1 * t2 / (t1 + t2);
    };

    effective_mass = compute_effective_quantity(self.get_property(ObstacleType::Properties::mass),
                                                other.get_property(ObstacleType::Properties::mass));
    effective_radius =
      compute_effective_quantity(self.get_property(ObstacleType::Properties::radius),
                                 other.get_property(ObstacleType::Properties::radius));

    effective_youngs_modulus = 0.5 * youngs_modulus / (1 - poisson_ratio * poisson_ratio);

    const number distance = self.get_location().distance(other.get_location());

    normal_vector  = (other.get_location() - self.get_location()) / distance;
    normal_overlap = (self.get_property(ObstacleType::Properties::radius) +
                      other.get_property(ObstacleType::Properties::radius)) -
                     distance;

    if constexpr (dim == 3)
      {
        relative_velocity.value =
          self.get_linear_velocity() - other.get_linear_velocity() +
          dealii::cross_product_3d(self.get_property(ObstacleType::Properties::radius) *
                                       self.get_angular_velocity() +
                                     other.get_property(ObstacleType::Properties::radius) *
                                       other.get_angular_velocity(),
                                   normal_vector);
      }
    else if constexpr (dim == 2)
      {
        relative_velocity.value =
          self.get_linear_velocity() - other.get_linear_velocity() +
          (self.get_property(ObstacleType::Properties::radius) * self.get_angular_velocity()[0] +
           other.get_property(ObstacleType::Properties::radius) * other.get_angular_velocity()[0]) *
            dealii::cross_product_2d(normal_vector);
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage("Dimension not supported!"));
      }

    relative_velocity.normal_component = (normal_vector * relative_velocity.value) * normal_vector;
    relative_velocity.tangential_component =
      relative_velocity.value - relative_velocity.normal_component;
  }

} // namespace MeltPoolDG

template struct MeltPoolDG::SphericalParticleContactData<double>;

template class MeltPoolDG::
  SphericalParticleContactForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::
  SphericalParticleContactForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::
  SphericalParticleContactForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;
