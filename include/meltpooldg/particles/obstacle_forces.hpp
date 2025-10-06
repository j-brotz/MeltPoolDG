
#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/particles/particle_accessor.h>

#include <memory>
#include <utility>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  class ObstacleField;

  /**
   * @brief Interface class for forces acting on obstacles using type erasure.
   *
   * This interface enables uniform handling of different force models by abstracting them through
   * type erasure. Any force acting on the obstacles in an obstacle field must conform to this
   * interface.
   *
   * The only requirement for a class implementing a force model is to provide a member function
   * named
   * @p add_force_to_obstacles() that:
   * - Takes a @p MeltPoolDG::ObstacleField as its argument. This object provides
   *   access to all necessary data describing the obstacles. Also the forces are added to the
   *   corresponing obstacles in this field.
   */
  template <int dim, typename number, typename ObstacleType>
  struct ObstacleForce
  {
  private:
    struct ObstacleForceConcept
    {
      virtual ~ObstacleForceConcept() = default;

      virtual void
      add_force_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const = 0;
    };

    template <typename ObstacleForceType>
    struct ObstacleForceModel final : public ObstacleForceConcept
    {
      explicit ObstacleForceModel(ObstacleForceType &&obstacle_force)
        : obstacle_force(std::move(obstacle_force))
      {}

      void
      add_force_to_obstacles(
        ObstacleField<dim, number, ObstacleType> &obstacle_field) const override
      {
        obstacle_force.add_force_to_obstacles(obstacle_field);
      }

    private:
      const ObstacleForceType obstacle_force;
    };

    std::unique_ptr<ObstacleForceConcept> obstacle_force_pimpl;

  public:
    template <typename ObstacleForceType>
    explicit ObstacleForce(ObstacleForceType &&obstacle_force)
      : obstacle_force_pimpl(
          std::make_unique<ObstacleForceModel<ObstacleForceType>>(std::move(obstacle_force)))
    {}

    void
    add_force_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
    {
      obstacle_force_pimpl->add_force_to_obstacles(obstacle_field);
    }
  };

  /**
   * @brief Computes the gravitational force acting on an obstacle.
   *
   * @note The obstacle type must define a static method `get_property(...)` and an enum member
   * `Properties::mass` for retrieving the mass of the obstacle.
   */
  template <int dim, typename number, typename ObstacleType>
  struct ObstacleGravitationalForce
  {
  public:
    /**
     * @brief Constructs the gravitational force object with a given gravitational constant.
     *
     * @param gravitational_constant The gravitational constant to use in force computation.
     */
    explicit ObstacleGravitationalForce(const number gravitational_constant)
      : gravitational_constant(gravitational_constant)
    {}

    /**
     * @brief Computes the gravitational force acting on all obstacles in the given obstacle
     * field and adds the force contribution to the corresponding obstacle force property.
     *
     * The force is computed as:
     * F = -g * m
     * where is the gravitational constant and is the mass of the obstacle. The force is
     * applied in the negative direction of the last axis.
     *
     * @param obstacle_field The obstacle field containing all particles on which the
     * gravitational forces are to be computed.
     */
    void
    add_force_to_obstacles(const ObstacleField<dim, number, ObstacleType> &obstacle_field) const
    {
      for (dealii::Particles::ParticleAccessor<dim> obstacle :
           obstacle_field.get_particle_handler())
        {
          dealii::Tensor<1, dim, number> force;
          force[dim - 1] = -gravitational_constant *
                           ObstacleType::get_property(obstacle, ObstacleType::Properties::mass);
          ObstacleType::add_force(force, obstacle);
        }
    }

  private:
    const number gravitational_constant;
  };
} // namespace MeltPoolDG
