#pragma once

#include <deal.II/base/tensor.h>

#include <deal.II/particles/particle_accessor.h>
#include <deal.II/particles/property_pool.h>

#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/particle.hpp>

#include <functional>

namespace MeltPoolDG
{
  /**
   * This class acts as a lightweight wrapper around deal.II's ParticleAccessor
   * or PropertyPool handles. It allows read/write access to particle properties
   * (velocity, angular velocity, forces, etc.) and to the particle's spatial location.
   */
  template <int dim, typename number>
  class DEMParticleAccessor
  {
  public:
    /**
     * Construct accessor from a deal.II ParticleAccessor.
     *
     * @param particle Reference to a deal.II particle accessor.
     */
    DEMParticleAccessor(dealii::Particles::ParticleAccessor<dim> &particle);

    /**
     * @brief Construct accessor from a PropertyPool handle.
     *
     * @param property_pool Reference to the deal.II property pool.
     * @param handle Handle/index of the particle within the property pool.
     */
    DEMParticleAccessor(
      dealii::Particles::PropertyPool<dim>                       &property_pool,
      const typename dealii::Particles::PropertyPool<dim>::Handle handle) noexcept;

    /**
     * Copy constructor.
     */
    DEMParticleAccessor(const DEMParticleAccessor &other) noexcept = default;

    /**
     * Move constructor.
     */
    DEMParticleAccessor(DEMParticleAccessor &&other) noexcept = default;

    /**
     * Move assignment operator.
     */
    DEMParticleAccessor &
    operator=(DEMParticleAccessor &&other) noexcept = default;

    /**
     * Copy assignment operator.
     */
    DEMParticleAccessor &
    operator=(const DEMParticleAccessor &other) noexcept = default;

    /**
     * Returns a reference to the location of the particle.
     *
     * @return Reference to the particle location.
     */
    dealii::Point<dim, number> &
    get_location();

    /**
     * Same as above but for const access.
     */
    const dealii::Point<dim, number> &
    get_location() const;

    /**
     * Returns the linear velocity of the given particle, i.e., the translational velocity at the
     * particle center of mass.
     *
     * @return A tensor representing the linear velocity of the particle.
     */
    dealii::Tensor<1, dim, number>
    get_linear_velocity() const;

    number &
    linear_velocity(const unsigned int dimension)
    {
      AssertIndexRange(dimension, dim);
      return properties[SphericalParticle<dim, number>::Properties::velocity + dimension];
    }

    number &
    linear_acceleration(const unsigned int dimension)
    {
      AssertIndexRange(dimension, dim);
      return properties[SphericalParticle<dim, number>::Properties::acceleration + dimension];
    }

    number &
    angular_velocity(const unsigned int dimension)
    {
      AssertIndexRange(dimension, axial_dim<dim>);
      return properties[SphericalParticle<dim, number>::Properties::angular_velocity + dimension];
    }

    number &
    angular_acceleration(const unsigned int dimension)
    {
      AssertIndexRange(dimension, axial_dim<dim>);
      return properties[SphericalParticle<dim, number>::Properties::angular_acceleration +
                        dimension];
    }

    /**
     * Returns the angular velocity of the given particle.
     *
     * @return A tensor representing the angular velocity of the particle.
     */
    dealii::Tensor<1, axial_dim<dim>, number>
    get_angular_velocity() const;

    /**
     * Sets the specified force vector to the particle by assigning its components into the
     * particle's property array.
     *
     * @param force The force vector to assign to the particle.
     */
    void
    set_force(const dealii::Tensor<1, dim, number> &force);

    /**
     * Accumulates the specified force vector to the particle by adding its components into the
     * particle's property array.
     *
     * @param force The force vector to be added.
     */
    void
    add_force(const dealii::Tensor<1, dim, number> &force);

    dealii::Tensor<1, dim, number>
    get_force() const
    {
      dealii::Tensor<1, dim, number> force;
      for (int dimension = 0; dimension < dim; ++dimension)
        force[dimension] =
          properties[SphericalParticle<dim, number>::Properties::force + dimension];
      return force;
    }

    number &
    force(const unsigned int dimension)
    {
      AssertIndexRange(dimension, dim);
      return properties[SphericalParticle<dim, number>::Properties::force + dimension];
    }

    /**
     * Sets the specified torque vector to the particle by assigning its components into the
     * particle's property array.
     *
     * @param torque The torque vector to assign to the particle.
     */
    void
    set_torque(const dealii::Tensor<1, axial_dim<dim>, number> &torque);

    dealii::Tensor<1, axial_dim<dim>, number>
    get_torque() const
    {
      dealii::Tensor<1, axial_dim<dim>, number> torque;
      for (int dimension = 0; dimension < axial_dim<dim>; ++dimension)
        torque[dimension] =
          properties[SphericalParticle<dim, number>::Properties::torque + dimension];
      return torque;
    }

    number &
    torque(const unsigned int dimension)
    {
      AssertIndexRange(dimension, axial_dim<dim>);
      return properties[SphericalParticle<dim, number>::Properties::torque + dimension];
    }

    /**
     * Accumulates the specified torque vector to the particle by adding its components into the
     * internal torque vector.
     *
     * @param torque The torque vector to be added.
     */
    void
    add_torque(const dealii::Tensor<1, axial_dim<dim>, number> &torque);

    /**
     * Returns the unique identifier of the particle.
     *
     * @return The particle ID.
     */
    const number &
    id() const;

    /**
     * Returns the radius the particle.
     *
     * @return The particle radius.
     */
    const number &
    radius() const;

    /**
     * This function returns the value of a single-valued property stored in the given particle.
     *
     * @param property The property to retrieve.
     * @return The value of the specified property.
     *
     * @note This function should be used only for scalar properties. For vector-valued properties
     * such as translational or angular velocity, use the functions @p get_velocity() and
     * @p get_angular_velocity().
     */
    number &
    get_property(const typename SphericalParticle<dim, number>::Properties property);

    /**
     * Same as above but for const access.
     */
    const number &
    get_property(const typename SphericalParticle<dim, number>::Properties property) const;

    /**
     * Return the surrounding cell of the particle center location. This is a wrapper around the
     * same function in the deal.II ParticleAccessor, which is used internally to keep track of the
     * cell in which the particle is located.
     */
    typename dealii::Triangulation<dim>::cell_iterator
    get_surrounding_cell() const
    {
      return surrounding_cell;
    }

  private:
    /// Reference to the particle location.
    std::reference_wrapper<dealii::Point<dim, number>> location;

    /// View to the particle properties (velocity, force etc.).
    dealii::ArrayView<number> properties;

    ///
    typename dealii::Triangulation<dim>::cell_iterator surrounding_cell;
  };


  template <int dim, typename number>
  DEMParticleAccessor<dim, number>::DEMParticleAccessor(
    dealii::Particles::ParticleAccessor<dim> &particle)
    : location(particle.get_location())
    , properties(particle.get_properties())
    , surrounding_cell(particle.get_surrounding_cell())
  {}

  template <int dim, typename number>
  DEMParticleAccessor<dim, number>::DEMParticleAccessor(
    dealii::Particles::PropertyPool<dim>                       &property_pool,
    const typename dealii::Particles::PropertyPool<dim>::Handle handle) noexcept
    : location(property_pool.get_location(handle))
    , properties(property_pool.get_properties(handle))
  {}

  template <int dim, typename number>
  dealii::Point<dim, number> &
  DEMParticleAccessor<dim, number>::get_location()
  {
    return location;
  }

  template <int dim, typename number>
  const dealii::Point<dim, number> &
  DEMParticleAccessor<dim, number>::get_location() const
  {
    return location;
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim, number>
  DEMParticleAccessor<dim, number>::get_linear_velocity() const
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    dealii::Tensor<1, dim, number> velocity;
    for (int dimension = 0; dimension < dim; ++dimension)
      velocity[dimension] =
        properties[SphericalParticle<dim, number>::Properties::velocity + dimension];
    return velocity;
  }

  template <int dim, typename number>
  dealii::Tensor<1, axial_dim<dim>, number>
  DEMParticleAccessor<dim, number>::get_angular_velocity() const
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    dealii::Tensor<1, axial_dim<dim>, number> velocity;
    for (int dimension = 0; dimension < axial_dim<dim>; ++dimension)
      velocity[dimension] =
        properties[SphericalParticle<dim, number>::Properties::angular_velocity + dimension];
    return velocity;
  }

  template <int dim, typename number>
  void
  DEMParticleAccessor<dim, number>::set_force(const dealii::Tensor<1, dim, number> &force)
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    for (int dimension = 0; dimension < dim; ++dimension)
      properties[SphericalParticle<dim, number>::Properties::force + dimension] = force[dimension];
  }

  template <int dim, typename number>
  void
  DEMParticleAccessor<dim, number>::add_force(const dealii::Tensor<1, dim, number> &force)
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    for (int dimension = 0; dimension < dim; ++dimension)
      properties[SphericalParticle<dim, number>::Properties::force + dimension] += force[dimension];
  }

  template <int dim, typename number>
  void
  DEMParticleAccessor<dim, number>::set_torque(
    const dealii::Tensor<1, axial_dim<dim>, number> &torque)
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    for (int dimension = 0; dimension < axial_dim<dim>; ++dimension)
      properties[SphericalParticle<dim, number>::Properties::torque + dimension] =
        torque[dimension];
  }

  template <int dim, typename number>
  void
  DEMParticleAccessor<dim, number>::add_torque(
    const dealii::Tensor<1, axial_dim<dim>, number> &torque)
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    for (int dimension = 0; dimension < axial_dim<dim>; ++dimension)
      properties[SphericalParticle<dim, number>::Properties::torque + dimension] +=
        torque[dimension];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::id() const
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    return properties[SphericalParticle<dim, number>::Properties::particle_id];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::radius() const
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    return properties[SphericalParticle<dim, number>::Properties::radius];
  }

  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::get_property(
    const typename SphericalParticle<dim, number>::Properties property)
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    return properties[property];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::get_property(
    const typename SphericalParticle<dim, number>::Properties property) const
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    return properties[property];
  }
} // namespace MeltPoolDG