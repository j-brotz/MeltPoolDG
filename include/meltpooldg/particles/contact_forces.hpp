#pragma once

#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/particles/particle_accessor.h>

#include "meltpooldg/utilities/dealii_tensor.hpp"
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <cmath>
#include <numbers>


namespace MeltPoolDG
{
  template <typename number>
  struct SphericalParticleContactData
  {
    number restitution_coefficient;
    number sliding_friction_coefficient;
    number impact_relative_velocity;

    struct MaterialData
    {
      number youngs_modulus;
      number shear_modulus;

      number poisson_ratio;
    };

    MaterialData particle;
    MaterialData wall; // assumed to be infinitely stiff!

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("spherical particle contact force");
      {
        prm.add_parameter("sliding friction coefficient", sliding_friction_coefficient);
        prm.add_parameter("restitution coefficient", restitution_coefficient);
        prm.add_parameter("impact relative velocity", impact_relative_velocity);
        prm.enter_subsection("particle");
        {
          prm.add_parameter("youngs modulus", particle.youngs_modulus);
          prm.add_parameter("poisson ratio", particle.poisson_ratio);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    void
    post()
    {
      particle.shear_modulus = 1. / (2. * (1. + particle.poisson_ratio)) * particle.youngs_modulus;
    }
  };

  template <int dim, typename number>
  struct ContactWall
  {
    dealii::Functions::SignedDistance::Plane<dim> signed_distance;
    dealii::Tensor<1, dim, number>                unit_normal;
  };


  template <int dim, typename number, typename ObstacleType>
  struct TempContactQuantities
  {
    TempContactQuantities(dealii::Particles::ParticleAccessor<dim>   &self_particle,
                          const dealii::Particles::PropertyPool<dim> &property_pool,
                          const unsigned                              other_handle,
                          const number                                particle_youngs_modulus,
                          const number                                particle_poisson_ratio)
      : self(self_particle, particle_youngs_modulus, particle_poisson_ratio)
      , other(property_pool, other_handle, particle_youngs_modulus, particle_poisson_ratio)
      , contact_configuration(self, other)
    {}

    TempContactQuantities(dealii ::Particles::ParticleAccessor<dim> &particle,
                          const number                               particle_youngs_modulus,
                          const number                               particle_poisson_ratio,
                          ContactWall<dim, number>                   wall)
      : self(particle, particle_youngs_modulus, particle_poisson_ratio)
      , contact_configuration(self, wall)
    {}

    struct ParticleInfo
    {
      ParticleInfo() = default;

      ParticleInfo(const dealii::Particles::ParticleAccessor<dim> &particle,
                   const number                                    youngs_modulus,
                   const number                                    poisson_ratio)
        : radius(ObstacleType::get_property(particle, ObstacleType::Properties::radius))
        , mass(ObstacleType::get_property(particle, ObstacleType::Properties::mass))
        , youngs_modulus(youngs_modulus)
        , shear_modulus(1. / (2. * (1. + poisson_ratio)) * youngs_modulus)
        , poisson_ratio(poisson_ratio)
        , id(ObstacleType::get_property(particle, ObstacleType::Properties::particle_id))
        , location(particle.get_location())
        , linear_velocity(ObstacleType::template get_velocity<number>(particle))
        , angular_velocity(ObstacleType::get_angular_velocity(particle))
      {}

      ParticleInfo(const dealii::Particles::PropertyPool<dim> &property_pool,
                   const unsigned                              handle,
                   const number                                youngs_modulus,
                   const number                                poisson_ratio)
        : radius(
            ObstacleType::get_property(property_pool, handle, ObstacleType::Properties::radius))
        , mass(ObstacleType::get_property(property_pool, handle, ObstacleType::Properties::mass))
        , youngs_modulus(youngs_modulus)
        , shear_modulus(1. / (2. * (1. + poisson_ratio)) * youngs_modulus)
        , poisson_ratio(poisson_ratio)
        , id(ObstacleType::get_property(property_pool,
                                        handle,
                                        ObstacleType::Properties::particle_id))
        , location(property_pool.get_location(handle))
        , linear_velocity(ObstacleType::template get_velocity<number>(property_pool, handle))
        , angular_velocity(ObstacleType::get_angular_velocity(property_pool, handle))
      {}

      number radius;
      number mass;
      number youngs_modulus;
      number shear_modulus;
      number poisson_ratio;

      int id;

      dealii::Point<dim, number>                location;
      dealii::Tensor<1, dim, number>            linear_velocity;
      dealii::Tensor<1, axial_dim<dim>, number> angular_velocity;
    };

    ParticleInfo self;
    ParticleInfo other;

    struct ContactConfiguration
    {
      // constructor for particle-particle contact
      ContactConfiguration(ParticleInfo self, ParticleInfo other)
      {
        auto compute_effective_quantity = [](number t1, number t2) -> number {
          return t1 * t2 / (t1 + t2);
        };

        effective_mass   = compute_effective_quantity(self.mass, other.mass);
        effective_radius = compute_effective_quantity(self.radius, other.radius);

        Assert(
          self.youngs_modulus == other.youngs_modulus,
          dealii::ExcMessage(
            "Only particles with same Young's modulus are supported for particle-particle contact."));

        effective_youngs_modulus =
          0.5 * self.youngs_modulus / (1 - self.poisson_ratio * self.poisson_ratio);

        effective_shear_modulus = 0.5 * self.shear_modulus / ((2. - self.poisson_ratio));

        const number distance = self.location.distance(other.location);

        normal_vector  = (other.location - self.location) / distance;
        normal_overlap = (self.radius + other.radius) - distance;

        if constexpr (dim == 3)
          {
            // TODO: do we need the correction for the normal gap here?
            relative_velocity.value =
              self.linear_velocity - other.linear_velocity +
              dealii::cross_product_3d(self.radius * self.angular_velocity +
                                         other.radius * other.angular_velocity,
                                       normal_vector);
          }
        else if constexpr (dim == 2)
          {
            relative_velocity.value =
              self.linear_velocity - other.linear_velocity +
              (self.radius * self.angular_velocity[0] + other.radius * other.angular_velocity[0]) *
                dealii::cross_product_2d(normal_vector);
          }
        else
          {
            AssertThrow(false, dealii::ExcMessage("Dimension not supported!"));
          }

        relative_velocity.normal_component =
          (normal_vector * relative_velocity.value) * normal_vector;
        relative_velocity.tangential_component =
          relative_velocity.value - relative_velocity.normal_component;
      }

      // constructor for particle-wall contact
      ContactConfiguration(ParticleInfo particle, ContactWall<dim, number> wall)
      {
        // effective quantities
        effective_mass   = particle.mass;
        effective_radius = particle.radius;
        effective_youngs_modulus =
          particle.youngs_modulus / (1 - particle.poisson_ratio * particle.poisson_ratio);
        effective_shear_modulus = 2 * particle.shear_modulus / ((2. - particle.poisson_ratio));

        // normal direction config
        normal_overlap = particle.radius - wall.signed_distance.value(particle.location);
        normal_vector  = -wall.unit_normal;

        // relative velocity
        if constexpr (dim == 3)
          {
            // TODO: do we need the correction for the normal gap here?
            relative_velocity.value =
              particle.linear_velocity +
              dealii::cross_product_3d(particle.radius * particle.angular_velocity, normal_vector);
          }
        else if constexpr (dim == 2)
          {
            relative_velocity.value =
              particle.linear_velocity + particle.radius * particle.angular_velocity[0] *
                                           dealii::cross_product_2d(normal_vector);
          }
        else
          {
            AssertThrow(false, dealii::ExcMessage("Dimension not supported!"));
          }

        relative_velocity.normal_component =
          (normal_vector * relative_velocity.value) * normal_vector;
        relative_velocity.tangential_component =
          relative_velocity.value - relative_velocity.normal_component;
      }

      number effective_mass;
      number effective_radius;
      number effective_shear_modulus;
      number effective_youngs_modulus;

      dealii::Tensor<1, dim, number> normal_vector;
      number                         normal_overlap;

      struct
      {
        dealii::Tensor<1, dim, number> value;
        dealii::Tensor<1, dim, number> normal_component;
        dealii::Tensor<1, dim, number> tangential_component;
      } relative_velocity;

    } contact_configuration;
  };


  template <int dim, typename number, typename ObstacleType>
  struct SphericalParticleContactForce
  {
  public:
    explicit SphericalParticleContactForce(
      const SphericalParticleContactData<number>  &contact_data,
      const TimeIntegration::TimeIterator<number> &time_iterator,
      std::vector<ContactWall<dim, number>>        walls = {})
      : contact_data(contact_data)
      , time_iterator(time_iterator)
      , damping_prefactor(compute_damping_prefactor(contact_data.restitution_coefficient))
      , walls(walls)
    {}

    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
    {
      const dealii::Particles::PropertyPool<dim> &global_particle_properties =
        obstacle_field.get_obstacle_data_structure().get_global_particle_properties();

      for (auto &obstacle : obstacle_field.get_particle_handler())
        {
          // particle-particle contact
          for (unsigned other_handle = 0;
               other_handle < global_particle_properties.n_registered_slots();
               ++other_handle)
            {
              if (ObstacleType::get_property(obstacle, ObstacleType::Properties::particle_id) ==
                  ObstacleType::get_property(global_particle_properties,
                                             other_handle,
                                             ObstacleType::Properties::particle_id))
                continue;

              TempContactQuantities<dim, number, ObstacleType> contact_configuration(
                obstacle,
                global_particle_properties,
                other_handle,
                contact_data.particle.youngs_modulus,
                contact_data.particle.poisson_ratio);

              if (contact_configuration.contact_configuration.normal_overlap >= 0)
                {
                  dealii::Tensor<1, dim, number> normal_force =
                    normal_contact_force(contact_configuration);

                  dealii::Tensor<1, dim, number> tangential_force =
                    tangential_contact_force(contact_configuration,
                                             normal_force,
                                             tangential_gap[{contact_configuration.self.id,
                                                             contact_configuration.other.id}]);

                  dealii::Tensor<1, axial_dim<dim>, number> rolling_resistance =
                    compute_rolling_resistance_momentum(contact_configuration, normal_force, false);

                  ObstacleType::accumulate_torque(
                    compute_torque(tangential_force,
                                   contact_configuration.self.radius *
                                     contact_configuration.contact_configuration.normal_vector) +
                      rolling_resistance,
                    obstacle);

                  ObstacleType::accumulate_force(normal_force + tangential_force, obstacle);
                }
              else
                {
                  tangential_gap[{contact_configuration.self.id, contact_configuration.other.id}] =
                    dealii::Tensor<1, dim, number>();
                }
            }

          // particle wall contact
          for (auto &wall : walls)
            {
              TempContactQuantities<dim, number, ObstacleType> contact_configuration(
                obstacle,
                contact_data.particle.youngs_modulus,
                contact_data.particle.poisson_ratio,
                wall);

              if (contact_configuration.contact_configuration.normal_overlap >= 0)
                {
                  dealii::Tensor<1, dim, number> normal_force =
                    normal_contact_force(contact_configuration);

                  dealii::Tensor<1, dim, number> tangential_force =
                    tangential_contact_force(contact_configuration, normal_force, wall_tang_gap);

                  dealii::Tensor<1, axial_dim<dim>, number> rolling_resistance =
                    compute_rolling_resistance_momentum(contact_configuration, normal_force, true);
                  ObstacleType::accumulate_torque(
                    compute_torque(tangential_force,
                                   contact_configuration.self.radius *
                                     contact_configuration.contact_configuration.normal_vector) + rolling_resistance,
                    obstacle);

                  ObstacleType::accumulate_force(normal_force + tangential_force, obstacle);
                }
              else
                {
                  wall_tang_gap = dealii::Tensor<1, dim, number>();
                }
            }
        }
    }

  private:
    dealii::Tensor<1, dim, number>
    normal_contact_force(const TempContactQuantities<dim, number, ObstacleType> &data) const
    {
      const number normal_stiffness = 4. / 3. *
                                      data.contact_configuration.effective_youngs_modulus *
                                      std::sqrt(data.contact_configuration.effective_radius *
                                                data.contact_configuration.normal_overlap);

      const number normal_damping =
        damping_prefactor *
        std::sqrt(1.5 * normal_stiffness * data.contact_configuration.effective_mass);

      auto normal_force =
        -normal_stiffness * data.contact_configuration.normal_overlap *
          data.contact_configuration.normal_vector -
        normal_damping * data.contact_configuration.relative_velocity.normal_component;

      return normal_force;
    }

    dealii::Tensor<1, dim, number>
    tangential_contact_force(const TempContactQuantities<dim, number, ObstacleType> &data,
                             const dealii::Tensor<1, dim, number>                   &normal_force,
                             dealii::Tensor<1, dim, number> &tangential_gap_vec) const
    {
      if (data.contact_configuration.relative_velocity.tangential_component.norm() == 0 and
          tangential_gap_vec.norm() == 0)
        return dealii::Tensor<1, dim, number>();

      // tangential stiffness and damping
      number tangential_stiffness = 8. * data.contact_configuration.effective_shear_modulus *
                                    std::sqrt(data.contact_configuration.effective_radius *
                                              data.contact_configuration.normal_overlap);

      number tangential_damping =
        damping_prefactor *
        std::sqrt(tangential_stiffness * data.contact_configuration.effective_mass);

      // Update tangential force
      tangential_gap_vec += time_iterator.get_current_time_increment() *
                            data.contact_configuration.relative_velocity.tangential_component;

      dealii::Tensor<1, dim, number> sticking_contact_force =
        -tangential_stiffness * tangential_gap_vec -
        tangential_damping * data.contact_configuration.relative_velocity.tangential_component;

      // slip force
      const number slip_force = contact_data.sliding_friction_coefficient * normal_force.norm();

      if (slip_force < sticking_contact_force.norm())
        {
          auto direction = sticking_contact_force / sticking_contact_force.norm();
          dealii::Tensor<1, dim, number> limited_tangential_force = slip_force * direction;

          dealii::Tensor<1, dim, number> limited_elastic_tangential_force =
            limited_tangential_force +
            tangential_damping * data.contact_configuration.relative_velocity.tangential_component;

          tangential_gap_vec = -1. / tangential_stiffness * limited_elastic_tangential_force;


          return limited_tangential_force;
        }
      else
        {
          return sticking_contact_force;
        }
    }

    number
    compute_damping_prefactor(const number restitution_coefficient) const
    {
      return -2. * std::sqrt(5. / 6.) * std::log(restitution_coefficient) /
             std::sqrt(std::pow(std::log(restitution_coefficient), 2) +
                       std::pow(std::numbers::pi_v<number>, 2));
    }

    dealii::Tensor<1, axial_dim<dim>, number>
    compute_rolling_resistance_momentum(
      const TempContactQuantities<dim, number, ObstacleType> &data,
      const dealii::Tensor<1, dim, number>                   &normal_force,
      const bool                                              at_wall) const
    {
      dealii::Tensor<1, axial_dim<dim>, number> other_angular =
        at_wall ? dealii::Tensor<1, axial_dim<dim>, number>() : data.other.angular_velocity;
      number rolling_resistance_coefficient =
        (1 - contact_data.restitution_coefficient) /
        (1.15344 * std::pow(contact_data.impact_relative_velocity, 0.2)) *
        std::pow(contact_data.particle.youngs_modulus *
                   std::sqrt(data.contact_configuration.effective_radius * 0.5) /
                   (1 - contact_data.particle.poisson_ratio * contact_data.particle.poisson_ratio),
                 -0.2);

      dealii::Tensor<1, axial_dim<dim>, number> delta_angular_velocity;
      if constexpr (dim == 3)
        {
          delta_angular_velocity =
            matrix_vector_product<dim, dim, number>((identity<dim, number>()),
                                                    (data.self.angular_velocity-other_angular)) -
            matrix_vector_product<dim, dim, number>(
              (dyadic_product(data.contact_configuration.normal_vector,
                              data.contact_configuration.normal_vector)),
              (data.self.angular_velocity-other_angular));
        }
      if constexpr (dim == 2)
        {
        }

      return -rolling_resistance_coefficient * normal_force.norm() *
             data.contact_configuration.effective_radius * delta_angular_velocity;
    }

    SphericalParticleContactData<number> contact_data;

    const TimeIntegration::TimeIterator<number> &time_iterator;

    mutable dealii::Tensor<1, dim, number> wall_tang_gap;

    mutable std::map<std::pair<int, int>, dealii::Tensor<1, dim, number>> tangential_gap;

    const number damping_prefactor;

    std::vector<ContactWall<dim, number>> walls;
  };
} // namespace MeltPoolDG