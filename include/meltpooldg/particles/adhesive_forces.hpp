#pragma once

#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/time_integration/time_iterator.hpp>

#include <cmath>
#include <numbers>


namespace MeltPoolDG
{
  template <typename number>
  struct SphericalParticleAdhesiveForceData
  {
    number unit_surface_energy = 0.0001;

    number hamaker_constant = 40e-20;

    number vdw_force_cutoff_fraction = 0.01;

    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("spherical particle adhesive force");
      {
        prm.add_parameter("unit surface energy", unit_surface_energy);
        prm.add_parameter("hamaker constant", hamaker_constant);
        prm.add_parameter("vdw force cutoff fraction", vdw_force_cutoff_fraction);
      }
      prm.leave_subsection();
    }
  };

  template <int dim, typename number, typename ObstacleType>
  struct AdhesiveTempContactQuantities
  {
    AdhesiveTempContactQuantities(dealii::Particles::ParticleAccessor<dim>         &self_particle,
                                  const dealii::Particles::PropertyPool<dim>       &property_pool,
                                  const unsigned                                    other_handle,
                                  const SphericalParticleAdhesiveForceData<number> &data)
      : self(self_particle)
      , other(property_pool, other_handle)
      , contact_configuration(self, other, data)
    {}

    struct ParticleInfo
    {
      ParticleInfo() = default;

      ParticleInfo(const dealii::Particles::ParticleAccessor<dim> &particle)
        : radius(ObstacleType::get_property(particle, ObstacleType::Properties::radius))
        , id(ObstacleType::get_property(particle, ObstacleType::Properties::particle_id))
        , location(particle.get_location())
      {}

      ParticleInfo(const dealii::Particles::PropertyPool<dim> &property_pool, const unsigned handle)
        : radius(
            ObstacleType::get_property(property_pool, handle, ObstacleType::Properties::radius))
        , id(ObstacleType::get_property(property_pool,
                                        handle,
                                        ObstacleType::Properties::particle_id))
        , location(property_pool.get_location(handle))
      {}

      number radius;

      int id;

      dealii::Point<dim, number> location;
    };

    ParticleInfo self;
    ParticleInfo other;

    struct ContactConfiguration
    {
      ContactConfiguration(ParticleInfo                               self,
                           ParticleInfo                               other,
                           SphericalParticleAdhesiveForceData<number> data)
      {
        auto compute_effective_quantity = [](number t1, number t2) -> number {
          return t1 * t2 / (t1 + t2);
        };

        effective_radius = compute_effective_quantity(self.radius, other.radius);


        const number distance = self.location.distance(other.location);

        normal_vector  = (other.location - self.location) / distance;
        normal_overlap = distance - (self.radius + other.radius);

        pull_off_force = 4 * std::numbers::pi * data.unit_surface_energy * effective_radius;

        distance_van_der_waals_equals_pull_off =
          std::sqrt(effective_radius * data.hamaker_constant / (6.0 * pull_off_force));

        cut_off_radius =
          distance_van_der_waals_equals_pull_off / std::sqrt(data.vdw_force_cutoff_fraction);
      }

      number effective_radius;

      dealii::Tensor<1, dim, number> normal_vector;
      number                         normal_overlap;
      number                         cut_off_radius;
      number                         distance_van_der_waals_equals_pull_off;

      number pull_off_force;

    } contact_configuration;
  };


  template <int dim, typename number, typename ObstacleType>
  struct SphericalParticleAdhesiveForce
  {
  public:
    explicit SphericalParticleAdhesiveForce(
      const SphericalParticleAdhesiveForceData<number> &adhesive_force_data)
      : adhesive_force_data(adhesive_force_data)
    {}

    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
    {
      const dealii::Particles::PropertyPool<dim> &global_particle_properties =
        obstacle_field.get_obstacle_data_structure().get_global_particle_properties();

      for (auto &obstacle : obstacle_field.get_particle_handler())
        {
          for (unsigned other_handle = 0;
               other_handle < global_particle_properties.n_registered_slots();
               ++other_handle)
            {
              if (ObstacleType::get_property(obstacle, ObstacleType::Properties::particle_id) ==
                  ObstacleType::get_property(global_particle_properties,
                                             other_handle,
                                             ObstacleType::Properties::particle_id))
                continue;

              AdhesiveTempContactQuantities<dim, number, ObstacleType> contact_configuration(
                obstacle, global_particle_properties, other_handle, adhesive_force_data);

              dealii::Tensor<1, dim, number> adhesive_force;

              if (contact_configuration.contact_configuration.normal_overlap <=
                  contact_configuration.contact_configuration
                    .distance_van_der_waals_equals_pull_off)
                {
                  adhesive_force = -contact_configuration.contact_configuration.pull_off_force *
                                   contact_configuration.contact_configuration.normal_vector;
                }
              else if (contact_configuration.contact_configuration.normal_overlap <
                         contact_configuration.contact_configuration.cut_off_radius and
                       contact_configuration.contact_configuration.normal_overlap >
                         contact_configuration.contact_configuration
                           .distance_van_der_waals_equals_pull_off)
                {
                  adhesive_force =
                    adhesive_force_data.hamaker_constant *
                    contact_configuration.contact_configuration.effective_radius /
                    (6. * contact_configuration.contact_configuration.normal_overlap *
                     contact_configuration.contact_configuration.normal_overlap) *
                    contact_configuration.contact_configuration.normal_vector;
                }

              ObstacleType::accumulate_force(adhesive_force, obstacle);
            }
        }
    }

  private:
    SphericalParticleAdhesiveForceData<number> adhesive_force_data;
  };
} // namespace MeltPoolDG