
#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_accessor.h>

#include <meltpooldg/utilities/utility_functions.hpp>

#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG
{
  /**
   * @brief Class representing a finite-sized spherical particle, intended for use with the obstacle
   * field class.
   *
   * Unlike point particles, this class models particles with a physical extent (non-zero radius,
   * mass etc.), allowing them to behave as rigid spherical bodies. It integrates with the deal.II
   * ParticleHandler infrastructure.
   */
  template <int dim, typename number>
  class Particle
  {
  public:
    // Number of components of the angular velocity: one in 2D, three in 3D.
    static constexpr unsigned int size_angular_velocity = dim - 3 % dim;

    // Total number of scalar properties associated with each obstacle particle.
    // Breakdown:
    // - 2 * size_angular_velocity: angular velocity and angular acceleration
    // - 2 * dim: velocity and acceleration
    // - dim: (directional) force acting on the particle
    // - 5: radius, volume, density, mass, moment of intertia
    static constexpr unsigned int n_obstacle_properties =
      2 * size_angular_velocity + 2 * dim + dim + 5;

    /**
     * @brief Enum to define the index of each property in the particle's property vector.
     */
    enum Properties
    {
      velocity             = 0,
      acceleration         = dim,
      angular_velocity     = acceleration + dim,
      angular_acceleration = angular_velocity + size_angular_velocity,
      force                = angular_acceleration + size_angular_velocity,
      radius               = force + dim,
      volume,
      density,
      mass,
      moment_of_inertia,
    };

    /**
     * @brief Parses particle properties and location from a single CSV-formatted line. This
     * function reads a single line of CSV-formatted text containing data for one particle. It
     * extracts the particle's properties and its location, and returns them as a pair.
     *
     * @param input_data A string containing one CSV-formatted line representing a single particle's
     * data.
     * @return A pair consisting of a std::vector containing the particle's properties and a
     * dealii::Point representing the particle's location.
     */
    static std::pair<std::vector<number>, dealii::Point<dim, number>>
    read_state_input(const std::string &input_data);

    /**
     * @brief Returns the force vector acting on the given particle.
     *
     * @param particle The particle of interest.
     * @return A dealii::Tensor representing the force vector acting on the particle.
     */
    static dealii::Tensor<1, dim, number>
    get_force(const dealii::Particles::ParticleAccessor<dim> &particle);

    /**
     * @brief Assigns the specified force vector to the particle by writing its components into the
     * particle's property array.
     *
     * @param force The force vector to assign to the particle.
     * @param particle The particle to which the force will be applied.
     */
    static void
    set_force(const dealii::Tensor<1, dim, number>     &force,
              dealii::Particles::ParticleAccessor<dim> &particle);

    /**
     * @brief Returns the translational velocity vector of the specified particle, i.e., the
     * velocity at the particle center.
     *
     * @param particle The particle of interest.
     * @return The translational velocity vector at the particle center.
     */
    template <typename VectorizedArrayType>
    static dealii::Tensor<1, dim, VectorizedArrayType>
    get_velocity(const dealii::Particles::ParticleAccessor<dim> &particle);

    /**
     * @brief Retrieves the translational velocity of a specific particle from a property pool.
     *
     * This function returns the translational velocity of a particle, similar to the overload
     * above. However, instead of accessing properties of a single particle directly, it requires a
     * `PropertyPool` containing the properties of one or more particles. A handle must also be
     * provided to identify the specific particle within this pool.
     *
     * @param property_pool A reference to the property pool containing particle data.
     * @param handle A handle used to identify the particle of interest within the pool.
     * @return The translational velocity vector at the particle center.
     */
    template <typename VectorizedArrayType>
    static dealii::Tensor<1, dim, VectorizedArrayType>
    get_velocity(dealii::Particles::PropertyPool<dim>                 &property_pool,
                 typename dealii::Particles::PropertyPool<dim>::Handle handle);

    /**
     * @brief Computes the velocity of a particle at a user-defined location.
     *
     * Given a specific particle @p particle, this function calculates its velocity at a specified
     * point @p location, which does not necessarily have to be the particle center. The velocity
     * at the location is computed using the formula:
     *
     *     v + ω × r
     *
     * where:
     * - @p v is the translational velocity at the particle center,
     * - @p ω is the angular velocity,
     * - @p r is the vector from the particle center to the location of interest.
     *
     * @param particle The particle whose velocity is to be computed.
     * @param location The point at which to evaluate the velocity.
     * @return A tensor representing the velocity of the particle at the given location.
     */
    template <typename VectorizedArrayType>
    static dealii::Tensor<1, dim, VectorizedArrayType>
    get_velocity(const dealii::Particles::ParticleAccessor<dim> &particle,
                 const dealii::Point<dim, VectorizedArrayType>  &location);

    /**
     * @brief Computes the velocity of a particle at a specified location using a property pool.
     *
     * This function performs the same computation as the overload above, but instead of directly
     * accessing a particle object, it uses a `PropertyPool` and a corresponding @p handle to
     * identify the particle of interest. The velocity is evaluated at a user-defined location
     * @p location, not necessarily at the particle center.
     *
     * The velocity is computed as:
     *
     *     v + ω × r
     *
     * where:
     * - @p v is the translational velocity obtained from the property pool,
     * - @p ω is the angular velocity,
     * - @p r is the vector from the particle center to the evaluation point.
     *
     * @param property_pool The property pool containing the particle data.
     * @param handle The handle identifying the specific particle within the property pool.
     * @param location The point at which to evaluate the particle's velocity.
     * @return A tensor representing the velocity of the particle at the specified location.
     */
    template <typename VectorizedArrayType>
    static dealii::Tensor<1, dim, VectorizedArrayType>
    get_velocity(dealii::Particles::PropertyPool<dim>                 &property_pool,
                 typename dealii::Particles::PropertyPool<dim>::Handle handle,
                 const dealii::Point<dim, VectorizedArrayType>        &location);

    /**
     * @brief Returns the translational acceleration of the specified particle.
     *
     * This function retrieves the translational acceleration vector at the center of the given
     * particle @p particle.
     *
     * @param particle The particle whose center acceleration is to be returned.
     * @return A tensor representing the translational acceleration at the particle center.
     */
    static dealii::Tensor<1, dim, number>
    get_acceleration(const dealii::Particles::ParticleAccessor<dim> &particle);

    /**
     * @brief Returns the acceleration of a particle at a specified location.
     *
     * This function computes the particle's acceleration at a user-defined point @p location,
     * which may differ from the particle center. It includes both translational and rotational
     * contributions using the formula:
     *
     *     a + α × r
     *
     * where:
     * - @p a is the translational acceleration at the particle center,
     * - @p α is the angular acceleration,
     * - @p r is the vector from the particle center to the evaluation point.
     *
     * @param particle The particle whose acceleration is to be computed.
     * @param location The point at which to evaluate the acceleration.
     * @return A tensor representing the total acceleration at the specified location.
     */
    static dealii::Tensor<1, dim, number>
    get_acceleration(const dealii::Particles::ParticleAccessor<dim> &particle,
                     const dealii::Point<dim, number>               &location);

    /**
     * @brief Returns the angular velocity of the given particle.
     *
     * @param particle The particle whose angular velocity is to be returned.
     * @return A tensor representing the angular velocity of the particle.
     */
    static dealii::Tensor<1, size_angular_velocity, number>
    get_angular_velocity(const dealii::Particles::ParticleAccessor<dim> &particle);


    /**
     * @brief Returns the angular velocity of a particle identified by a handle in a property pool.
     *
     * @param property_pool The pool containing properties of one or more particles.
     * @param handle The handle identifying the specific particle within the property pool.
     * @return A tensor representing the angular velocity of the specified particle.
     */
    static dealii::Tensor<1, size_angular_velocity, number>
    get_angular_velocity(dealii::Particles::PropertyPool<dim>                 &property_pool,
                         typename dealii::Particles::PropertyPool<dim>::Handle handle);

    /**
     * @brief Returns the angular acceleration of a given particle.
     *
     * @param particle The particle whose angular acceleration is to be returned.
     * @return A tensor representing the angular acceleration of the particle.
     */
    static dealii::Tensor<1, size_angular_velocity, number>
    get_angular_acceleration(const dealii::Particles::ParticleAccessor<dim> &particle);

    /**
     * @brief Provides access to a specific single-valued property of a given particle.
     *
     * This function returns the value of a single-valued property stored in the given particle. It
     * should be used only for scalar properties.
     *
     * @param particle The particle from which to retrieve the property.
     * @param property The property to retrieve.
     * @return The value of the specified property.
     *
     * @note This function should be used only for scalar properties. For vector-valued properties
     * such as translational or angular velocity, use the functions @p get_velocity() and
     * @p get_angular_velocity(). The same applies to translational and angular acceleration.
     */
    static number
    get_property(const dealii::Particles::ParticleAccessor<dim> &particle, Properties property);

    /**
     * @brief Provides access to a specific single-valued property using a property pool and a
     * corresponding property handle.
     *
     * This function retrieves the value of a single-valued property for a particle identified by
     * the given handle within the provided property pool. Unlike the other overload, this version
     * does not require direct access to a particle object.
     *
     * @param property_pool The property pool containing data for multiple particles.
     * @param handle A handle identifying the specific particle within the property pool.
     * @param property The property to retrieve.
     * @return The value of the specified property.
     *
     * @note This function should be used only for scalar properties. For vector-valued properties
     * such as translational or angular velocity, use the functions @p get_velocity() and
     * @p get_angular_velocity(). The same applies to translational and angular acceleration.
     */
    static number
    get_property(dealii::Particles::PropertyPool<dim>                 &property_pool,
                 typename dealii::Particles::PropertyPool<dim>::Handle handle,
                 const Properties                                      property);

    /**
     * @brief Estimates whether a particle likely intersects or encloses a given cell.
     *
     * This function heuristically determines whether the specified particle @p particle at least
     * partially overlaps the given cell @p cell. Note that a return value of `true` does not
     * definitively confirm overlap but indicates a high likelihood based on the following logic:
     *
     * 1. If the center of the cell lies within the particle's radius, the function returns `true`.
     * 2. Otherwise, the function checks whether the distance between the particle center and the
     *    cell center is less than the sum of the particle radius and half the cell's diameter.
     *    If so, the function also returns `true`, though this is a conservative approximation.
     *
     * @param particle The particle accessor providing access to the particle's properties.
     * @param cell The finite element cell to be tested against the particle.
     * @return `true` if the particle likely overlaps the cell; otherwise, `false`.
     */
    static bool
    in_cell(const dealii::Particles::ParticleAccessor<dim> &particle,
            const dealii::CellAccessor<dim>                &cell)
    {
      return particle.get_location().distance(cell.center()) <
             get_property(particle, Properties::radius) + 0.5 * cell.diameter();
    }


    /**
     * @brief Estimates whether a particle identified in a property pool likely intersects a cell.
     *
     * This overload performs the same overlap estimation as the version above, but accesses the
     * particle's data indirectly via a property pool and a corresponding handle. The same heuristic
     * strategy is used:
     *
     * Check if the distance between the particle center and cell center is less than the sum  of
     * the particle radius and half the cell's diameter.
     *
     * @param property_pool The property pool containing particle data.
     * @param handle The handle identifying the particle in the property pool.
     * @param cell The finite element cell to test for potential overlap.
     * @return `true` if the particle likely overlaps the cell; otherwise, `false`.
     */
    static bool
    in_cell(dealii::Particles::PropertyPool<dim>                 &property_pool,
            typename dealii::Particles::PropertyPool<dim>::Handle handle,
            const dealii::CellAccessor<dim>                      &cell)
    {
      return property_pool.get_location(handle).distance(cell.center()) <
             get_property(property_pool, handle, Properties::radius) + 0.5 * cell.diameter();
    }
  };
} // namespace MeltPoolDG


template <int dim, typename number>
std::pair<std::vector<number>, dealii::Point<dim, number>>
MeltPoolDG::Particle<dim, number>::read_state_input(const std::string &input_data)
{
  dealii::Point<dim, number> particle_location;
  std::vector<number>        particle_properties(n_obstacle_properties, 0.0);
  std::string                temp;
  std::istringstream         data_string(input_data);
  // particle position
  for (unsigned int i = 0; i < dim; i++)
    {
      std::getline(data_string, temp, ',');
      particle_location[i] = std::stod(temp);
    }

  // particle velocity
  for (unsigned int i = Properties::velocity; i < Properties::acceleration; ++i)
    {
      std::getline(data_string, temp, ',');
      particle_properties[i] = std::stod(temp);
    }

  // particle angular velocity
  for (unsigned int i = Properties::angular_velocity; i < Properties::angular_acceleration; ++i)
    {
      std::getline(data_string, temp, ',');
      particle_properties[i] = std::stod(temp);
    }

  // density and radius
  std::getline(data_string, temp, ',');
  particle_properties[Properties::density] = std::stod(temp);
  std::getline(data_string, temp, ',');
  particle_properties[Properties::radius] = std::stod(temp);

  // Compute remaining variables
  if constexpr (dim == 3)
    {
      particle_properties[Properties::volume] =
        4.0 / 3.0 * M_PI * std::pow(particle_properties[Properties::radius], 3);
      particle_properties[Properties::mass] =
        particle_properties[Properties::volume] * particle_properties[Properties::density];
      particle_properties[Properties::moment_of_inertia] =
        0.4 * particle_properties[Properties::mass] *
        std::pow(particle_properties[Properties::radius], 2);
    }
  else if constexpr (dim == 2)
    {
      particle_properties[Properties::volume] =
        M_PI * std::pow(particle_properties[Properties::radius], 2);
      particle_properties[Properties::mass] =
        particle_properties[Properties::volume] * particle_properties[Properties::density];
      particle_properties[Properties::moment_of_inertia] =
        0.5 * particle_properties[Properties::mass] *
        std::pow(particle_properties[Properties::radius], 2);
    }

  return {particle_properties, particle_location};
}

template <int dim, typename number>
number
MeltPoolDG::Particle<dim, number>::get_property(
  const dealii::Particles::ParticleAccessor<dim> &particle,
  const Properties                                property)
{
  Assert(property >= Properties::radius,
         dealii::ExcMessage("For acces of paritcle velocities use the specific velocity getters!"));
  return particle.get_properties()[property];
}

template <int dim, typename number>
number
MeltPoolDG::Particle<dim, number>::get_property(
  dealii::Particles::PropertyPool<dim>                 &property_pool,
  typename dealii::Particles::PropertyPool<dim>::Handle handle,
  const Properties                                      property)
{
  Assert(property >= Properties::radius,
         dealii::ExcMessage("For acces of paritcle velocities use the specific velocity getters!"));
  return property_pool.get_properties(handle)[property];
}

template <int dim, typename number>
auto
MeltPoolDG::Particle<dim, number>::get_angular_velocity(
  const dealii::Particles::ParticleAccessor<dim> &particle)
  -> dealii::Tensor<1, size_angular_velocity, number>
{
  dealii::ArrayView<const number>                  properties = particle.get_properties();
  dealii::Tensor<1, size_angular_velocity, number> angular_velocity;
  for (unsigned int dimension = 0; dimension < size_angular_velocity; ++dimension)
    angular_velocity[dimension] = properties[Properties::angular_velocity + dimension];
  return angular_velocity;
}

template <int dim, typename number>
auto
MeltPoolDG::Particle<dim, number>::get_angular_velocity(
  dealii::Particles::PropertyPool<dim>                 &property_pool,
  typename dealii::Particles::PropertyPool<dim>::Handle handle)
  -> dealii::Tensor<1, size_angular_velocity, number>
{
  dealii::ArrayView<const number> properties = property_pool.get_properties(handle);
  dealii::Tensor<1, size_angular_velocity, number> angular_velocity;
  for (unsigned int dimension = 0; dimension < size_angular_velocity; ++dimension)
    angular_velocity[dimension] = properties[Properties::angular_velocity + dimension];
  return angular_velocity;
}

template <int dim, typename number>
auto
MeltPoolDG::Particle<dim, number>::get_angular_acceleration(
  const dealii::Particles::ParticleAccessor<dim> &particle)
  -> dealii::Tensor<1, size_angular_velocity, number>
{
  dealii::ArrayView<const number>                  properties = particle.get_properties();
  dealii::Tensor<1, size_angular_velocity, number> angular_acceleration;
  for (int dimension = 0; dimension < size_angular_velocity; ++dimension)
    angular_acceleration[dimension] = properties[Properties::angular_acceleration + dimension];
  return angular_acceleration;
}

template <int dim, typename number>
dealii::Tensor<1, dim, number>
MeltPoolDG::Particle<dim, number>::get_acceleration(
  const dealii::Particles::ParticleAccessor<dim> &particle)
{
  dealii::ArrayView<const number> properties = particle.get_properties();
  dealii::Tensor<1, dim, number>  acceleration;
  for (int dimension = 0; dimension < dim; ++dimension)
    acceleration[dimension] = properties[Properties::acceleration + dimension];
  return acceleration;
}

template <int dim, typename number>
void
MeltPoolDG::Particle<dim, number>::set_force(const dealii::Tensor<1, dim, number>     &force,
                                             dealii::Particles::ParticleAccessor<dim> &particle)
{
  dealii::ArrayView<number> properties = particle.get_properties();
  for (int dimension = 0; dimension < dim; ++dimension)
    properties[Properties::force + dimension] = force[dimension];
}

template <int dim, typename number>
dealii::Tensor<1, dim, number>
MeltPoolDG::Particle<dim, number>::get_force(
  const dealii::Particles::ParticleAccessor<dim> &particle)
{
  dealii::ArrayView<const number> properties = particle.get_properties();
  dealii::Tensor<1, dim, number>  force;
  for (int dimension = 0; dimension < dim; ++dimension)
    force[dimension] = properties[Properties::force + dimension];
  return force;
}

template <int dim, typename number>
template <typename VectorizedArrayType>
dealii::Tensor<1, dim, VectorizedArrayType>
MeltPoolDG::Particle<dim, number>::get_velocity(
  const dealii::Particles::ParticleAccessor<dim> &particle)
{
  dealii::ArrayView<const number> properties = particle.get_properties();
  dealii::Tensor<1, dim, number>  velocity;
  for (int dimension = 0; dimension < dim; ++dimension)
    velocity[dimension] = properties[Properties::velocity + dimension];
  return velocity;
}

template <int dim, typename number>
template <typename VectorizedArrayType>
dealii::Tensor<1, dim, VectorizedArrayType>
MeltPoolDG::Particle<dim, number>::get_velocity(
  dealii::Particles::PropertyPool<dim>                 &property_pool,
  typename dealii::Particles::PropertyPool<dim>::Handle handle)
{
  const dealii::ArrayView<const number> properties = property_pool.get_properties(handle);
  dealii::Tensor<1, dim, number>        velocity;
  for (int dimension = 0; dimension < dim; ++dimension)
    velocity[dimension] = properties[Properties::velocity + dimension];
  return velocity;
}

template <int dim, typename number>
template <typename VectorizedArrayType>
dealii::Tensor<1, dim, VectorizedArrayType>
MeltPoolDG::Particle<dim, number>::get_velocity(
  const dealii::Particles::ParticleAccessor<dim> &particle,
  const dealii::Point<dim, VectorizedArrayType>  &location)
{
  Assert(dim == 2 or dim == 3, dealii::ExcMessage("Invalid dimension!"));

  dealii::Point<dim, dealii::VectorizedArray<number>> vectorized_particle_location;
  for (auto i = 0; i < dim; ++i)
    vectorized_particle_location[i] = dealii::VectorizedArray<number>(particle.get_location()[i]);

  dealii::Tensor<1, dim, VectorizedArrayType> distance_to_center =
    location - vectorized_particle_location;
  if constexpr (dim == 2)
    {
      return dealii::cross_product_2d(get_angular_velocity(particle)[0] * distance_to_center) +
             get_velocity<VectorizedArrayType>(particle);
    }
  if constexpr (dim == 3)
    {
      return get_velocity<VectorizedArrayType>(particle) +
             dealii::cross_product_3d(get_angular_velocity(particle), distance_to_center);
    }
  AssertThrow(false, dealii::ExcInternalError());
}


template <int dim, typename number>
template <typename VectorizedArrayType>
dealii::Tensor<1, dim, VectorizedArrayType>
MeltPoolDG::Particle<dim, number>::get_velocity(
  dealii::Particles::PropertyPool<dim>                 &property_pool,
  typename dealii::Particles::PropertyPool<dim>::Handle handle,
  const dealii::Point<dim, VectorizedArrayType>        &location)
{
  Assert(dim == 2 or dim == 3, dealii::ExcMessage("Invalid dimension!"));

  dealii::Point<dim, dealii::VectorizedArray<number>> vectorized_particle_location;
  for (auto i = 0; i < dim; ++i)
    vectorized_particle_location[i] =
      dealii::VectorizedArray<number>(property_pool.get_location(handle)[i]);

  dealii::Tensor<1, dim, VectorizedArrayType> distance_to_center =
    location - vectorized_particle_location;
  if constexpr (dim == 2)
    {
      return dealii::cross_product_2d(get_angular_velocity(property_pool, handle)[0] *
                                      distance_to_center) +
             get_velocity<VectorizedArrayType>(property_pool, handle);
    }
  if constexpr (dim == 3)
    {
      return get_velocity<VectorizedArrayType>(property_pool, handle) +
             dealii::cross_product_3d(get_angular_velocity(property_pool, handle),
                                      distance_to_center);
    }
  AssertThrow(false, dealii::ExcInternalError());
}

template <int dim, typename number>
dealii::Tensor<1, dim, number>
MeltPoolDG::Particle<dim, number>::get_acceleration(
  const dealii::Particles::ParticleAccessor<dim> &particle,
  const dealii::Point<dim, number>               &location)
{
  Assert(dim == 2 or dim == 3, dealii::ExcMessage("Invalid dimension!"));

  dealii::Tensor<1, dim, number> distance_to_center = location - particle.get_location();
  if constexpr (dim == 2)
    {
      return dealii::cross_product_2d(get_angular_acceleration(particle) * distance_to_center) +
             get_acceleration(particle);
    }
  if constexpr (dim == 3)
    {
      return get_acceleration(particle) +
             dealii::cross_product_3d(get_angular_acceleration(particle), distance_to_center);
    }
  AssertThrow(false, dealii::ExcInternalError());
}
