
#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/particles/obstacle_data_structure.hpp>

#include <memory>
#include <utility>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  class ObstacleField;

  /**
   * @brief Interface class for loads acting on obstacles using type erasure.
   *
   * This interface enables uniform handling of different load models by abstracting them through
   * type erasure. Any load acting on the obstacles in an obstacle field must conform to this
   * interface. Hereby a load can be both forces and torques.
   *
   * The only requirement for a class implementing a load model is to provide a member function
   * named
   * @p add_load_to_obstacles() that:
   * - Takes a @p MeltPoolDG::ObstacleField as its argument. This object provides
   *   access to all necessary data describing the obstacles. Also the loads are added to the
   *   corresponding obstacles in this field.
   */
  template <int dim, typename number, typename ObstacleType>
  struct ObstacleLoad
  {
  private:
    struct ObstacleLoadConcept
    {
      virtual ~ObstacleLoadConcept() = default;

      virtual void
      add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const = 0;
    };

    template <typename ObstacleLoadType>
    struct ObstacleLoadModel final : public ObstacleLoadConcept
    {
      explicit ObstacleLoadModel(ObstacleLoadType &&obstacle_load)
        : obstacle_load(std::move(obstacle_load))
      {}

      void
      add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const override
      {
        obstacle_load.add_load_to_obstacles(obstacle_field);
      }

    private:
      const ObstacleLoadType obstacle_load;
    };

    std::unique_ptr<ObstacleLoadConcept> obstacle_load_pimpl;

  public:
    template <typename ObstacleLoadType>
    explicit ObstacleLoad(ObstacleLoadType &&obstacle_load)
      : obstacle_load_pimpl(
          std::make_unique<ObstacleLoadModel<ObstacleLoadType>>(std::move(obstacle_load)))
    {}

    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
    {
      obstacle_load_pimpl->add_load_to_obstacles(obstacle_field);
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
     * \[
     * F = -g * m
     * \]
     * where is the gravitational constant and is the mass of the obstacle. The force is
     * applied in the negative direction of the last axis.
     *
     * @param obstacle_field The obstacle field containing all particles on which the
     * gravitational forces are to be computed.
     */
    void
    add_load_to_obstacles(ObstacleField<dim, number, ObstacleType> &obstacle_field) const
    {
      for (MeltPoolDG::DEMParticleAccessor<dim, double> &particle :
           obstacle_field.locally_owned_particle_range())
        {
          dealii::Tensor<1, dim, number> force;
          force[dim - 1] =
            -gravitational_constant * particle.get_property(ObstacleType::Properties::mass);
          particle.add_force(force);
        }
    }

  private:
    const number gravitational_constant;
  };
} // namespace MeltPoolDG
