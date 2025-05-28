
#pragma once

#include <deal.II/base/tensor.h>

#include <memory>
#include <utility>

/**
 * @brief Interface class for forces acting on obstacles using type erasure.
 *
 * This interface enables uniform handling of different force models by abstracting them through
 * type erasure. Any force acting on an obstacle must conform to this interface.
 *
 * The only requirement for a class implementing a force model is to provide a member function named
 * @p compute_force_on_obstacle() that:
 * - Takes a @p dealii::Particles::ParticleAccessor<dim> as its argument. This object provides
 *   access to all necessary data describing the obstacle.
 * - Returns a @p dealii::Tensor<1, dim> representing the vectorial force acting on the obstacle.
 */
template <int dim, typename number, typename ObstacleType>
struct ObstacleForce
{
private:
  struct ObstacleForceConcept
  {
    virtual ~ObstacleForceConcept() = default;

    virtual dealii::Tensor<1, dim, number>
    compute_force_on_obstacle(const dealii::Particles::ParticleAccessor<dim> &obstacle) const = 0;
  };

  template <typename ObstacleForceType>
  struct ObstacleForceModel final : public ObstacleForceConcept
  {
    explicit ObstacleForceModel(ObstacleForceType &&obstacle_force)
      : obstacle_force(std::move(obstacle_force))
    {}

    dealii::Tensor<1, dim, number>
    compute_force_on_obstacle(
      const dealii::Particles::ParticleAccessor<dim> &obstacle) const override
    {
      return obstacle_force.compute_force_on_obstacle(obstacle);
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

  dealii::Tensor<1, dim, number>
  compute_force_on_obstacle(const dealii::Particles::ParticleAccessor<dim> &obstacle) const
  {
    return obstacle_force_pimpl->compute_force_on_obstacle(obstacle);
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
   * @brief Computes the gravitational force acting on a given obstacle.
   *
   * The force is computed as:
   * F = -g * m
   * where is the gravitational constant and is the mass of the obstacle. The force is
   * applied in the negative direction of the last axis.
   *
   * @param obstacle The obstacle on which the gravitational force is to be computed.
   * @return A Tensor representing the gravitational force vector.
   */
  dealii::Tensor<1, dim, number>
  compute_force_on_obstacle(const dealii::Particles::ParticleAccessor<dim> &obstacle) const
  {
    dealii::Tensor<1, dim, number> force;
    force[dim - 1] = -gravitational_constant *
                     ObstacleType::get_property(obstacle, ObstacleType::Properties::mass);
    return force;
  }

private:
  const number gravitational_constant;
};
