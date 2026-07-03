#pragma once

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
     * Construct accessor from a location and properties view.
     *
     * @param location Reference to the particle location.
     * @param properties View to the particle properties.
     * @param particle_id The unique identifier for the particle.
     */
    DEMParticleAccessor(dealii::Point<dim, number>   &location,
                        dealii::ArrayView<number>     properties,
                        dealii::types::particle_index particle_id);

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
    linear_velocity() const;

    /**
     * Returns the linear velocity of the given particle in the specified dimension.
     *
     * @param dimension The dimension for which to return the velocity.
     * @return The linear velocity in the specified dimension.
     */
    number &
    linear_velocity(const unsigned int dimension);

    /**
     * Same as above but for const access.
     */
    const number &
    linear_velocity(const unsigned int dimension) const;

    /**
     * Returns the linear acceleration of the given particle in the specified dimension.
     *
     * @param dimension The dimension for which to return the acceleration.
     * @return The linear acceleration in the specified dimension.
     */
    number &
    linear_acceleration(const unsigned int dimension);

    /**
     * Returns the angular velocity of the given particle.
     *
     * @return A tensor representing the angular velocity of the particle.
     */
    dealii::Tensor<1, axial_dim<dim>, number>
    get_angular_velocity() const;

    /**
     * Returns the angular acceleration of the given particle in the specified dimension.
     *
     * @param dimension The dimension for which to return the acceleration.
     * @return The angular acceleration in the specified dimension.
     */
    number &
    angular_acceleration(const unsigned int dimension);

    /**
     * Returns the angular velocity of the given particle in the specified dimension.
     *
     * @param dimension The dimension for which to return the angular velocity.
     * @return The angular velocity in the specified dimension.
     */
    number &
    angular_velocity(const unsigned int dimension);

    /**
     * Same as above but for const access.
     */
    const number &
    angular_velocity(const unsigned int dimension) const;

    /**
     * Returns the angular velocity tensor of the given particle.
     *
     * @return A tensor representing the angular velocity of the particle.
     */
    dealii::Tensor<1, axial_dim<dim>, number>
    angular_velocity() const;

    /**
     * Returns the force acting on the given particle in the specified dimension.
     *
     * @param dimension The dimension for which to return the force.
     * @return The force in the specified dimension.
     */
    number &
    force(const unsigned int dimension);

    /**
     * Same as above but for const access.
     */
    const number &
    force(const unsigned int dimension) const;

    /**
     * Returns the force vector of the given particle.
     *
     * @return A tensor representing the force of the particle.
     */
    dealii::Tensor<1, dim, number>
    force() const;

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

    /**
     * Sets the specified torque vector to the particle by assigning its components into the
     * particle's property array.
     *
     * @param torque The torque vector to assign to the particle.
     */
    void
    set_torque(const dealii::Tensor<1, axial_dim<dim>, number> &torque);

    /**
     * Accumulates the specified torque vector to the particle by adding its components into the
     * internal torque vector.
     *
     * @param torque The torque vector to be added.
     */
    void
    add_torque(const dealii::Tensor<1, axial_dim<dim>, number> &torque);

    /**
     * Returns the torque acting on the given particle in the specified dimension.
     *
     * @param dimension The dimension for which to return the torque.
     * @return The torque in the specified dimension.
     */
    number &
    torque(const unsigned int dimension);

    /**
     * Same as above but for const access.
     */
    const number &
    torque(const unsigned int dimension) const;

    /**
     * Returns the torque vector of the given particle.
     *
     * @return A tensor representing the torque of the particle.
     */
    dealii::Tensor<1, axial_dim<dim>, number>
    torque() const;

    /**
     * Returns the unique identifier of the particle.
     *
     * @return The particle ID.
     */
    dealii::types::particle_index
    id() const;

    /**
     * Returns the rank local particle index of the particle, which can be used for array access.
     *
     * @return The rank local particle index.
     */
    dealii::types::particle_index
    local_id() const;

    /**
     * Returns the radius of the particle.
     *
     * @return The radius of the particle.
     */
    number &
    radius();

    /**
     * Same as above but for const access.
     */
    const number &
    radius() const;

    /**
     * Returns the mass of the particle.
     *
     * @return The mass of the particle.
     */
    number &
    mass();

    /**
     * Same as above but for const access.
     */
    const number &
    mass() const;

    /**
     * Returns the density of the particle.
     *
     * @return The density of the particle.
     */
    number &
    density();

    /**
     * Same as above but for const access.
     */
    const number &
    density() const;

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
     * Return the surrounding active cell of the particle center location. This function can only be
     * called for locally owned particles as the surrounding active cell might not be locally
     * available for ghost particles.
     */
    typename dealii::Triangulation<dim>::active_cell_iterator
    get_surrounding_cell() const;

  private:
    /// Reference to the particle location.
    std::reference_wrapper<dealii::Point<dim, number>> location;

    /// View to the particle properties (velocity, force etc.).
    dealii::ArrayView<number> properties;

    /// The surrounding active cell of the particle center location. This is only valid for locally
    /// owned particles.
    typename dealii::Triangulation<dim>::active_cell_iterator surrounding_cell;

    /// The global unique identifier of the particle.
    dealii::types::particle_index particle_id;

    /// The MPI rank local particle index which can be used e.g. for array access (only available
    /// for locally owned particles).
    dealii::types::particle_index local_particle_id = dealii::numbers::invalid_unsigned_int;
  };


  template <int dim, typename number>
  DEMParticleAccessor<dim, number>::DEMParticleAccessor(
    dealii::Particles::ParticleAccessor<dim> &particle)
    : location(particle.get_location())
    , properties(particle.get_properties())
    , surrounding_cell(particle.get_surrounding_cell())
    , particle_id(particle.get_id())
    , local_particle_id(particle.get_local_index())
  {}

  template <int dim, typename number>
  DEMParticleAccessor<dim, number>::DEMParticleAccessor(dealii::Point<dim, number>   &location,
                                                        dealii::ArrayView<number>     properties,
                                                        dealii::types::particle_index particle_id)
    : location(location)
    , properties(properties)
    , particle_id(particle_id)
  {}

  template <int dim, typename number>
  DEMParticleAccessor<dim, number>::DEMParticleAccessor(
    dealii::Particles::PropertyPool<dim>                       &property_pool,
    const typename dealii::Particles::PropertyPool<dim>::Handle handle) noexcept
    : location(property_pool.get_location(handle))
    , properties(property_pool.get_properties(handle))
    , particle_id(property_pool.get_id(handle))
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
  DEMParticleAccessor<dim, number>::linear_velocity() const
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    dealii::Tensor<1, dim, number> velocity;
    for (int dimension = 0; dimension < dim; ++dimension)
      velocity[dimension] =
        properties[SphericalParticle<dim, number>::Properties::velocity + dimension];
    return velocity;
  }

  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::linear_velocity(const unsigned int dimension)
  {
    AssertIndexRange(dimension, dim);
    return properties[SphericalParticle<dim, number>::Properties::velocity + dimension];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::linear_velocity(const unsigned int dimension) const
  {
    AssertIndexRange(dimension, dim);
    return properties[SphericalParticle<dim, number>::Properties::velocity + dimension];
  }


  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::linear_acceleration(const unsigned int dimension)
  {
    AssertIndexRange(dimension, dim);
    return properties[SphericalParticle<dim, number>::Properties::acceleration + dimension];
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
  number &
  DEMParticleAccessor<dim, number>::angular_acceleration(const unsigned int dimension)
  {
    AssertIndexRange(dimension, axial_dim<dim>);
    return properties[SphericalParticle<dim, number>::Properties::angular_acceleration + dimension];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::angular_velocity(const unsigned int dimension) const
  {
    AssertIndexRange(dimension, axial_dim<dim>);
    return properties[SphericalParticle<dim, number>::Properties::angular_velocity + dimension];
  }

  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::angular_velocity(const unsigned int dimension)
  {
    AssertIndexRange(dimension, axial_dim<dim>);
    return properties[SphericalParticle<dim, number>::Properties::angular_velocity + dimension];
  }

  template <int dim, typename number>
  dealii::Tensor<1, axial_dim<dim>, number>
  DEMParticleAccessor<dim, number>::angular_velocity() const
  {
    Assert(!properties.empty(), dealii::ExcInternalError());
    dealii::Tensor<1, axial_dim<dim>, number> angular_velocity_vector;
    for (int dimension = 0; dimension < axial_dim<dim>; ++dimension)
      {
        angular_velocity_vector[dimension] = angular_velocity(dimension);
      }
    return angular_velocity_vector;
  }

  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::force(const unsigned int dimension)
  {
    AssertIndexRange(dimension, dim);
    return properties[SphericalParticle<dim, number>::Properties::force + dimension];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::force(const unsigned int dimension) const
  {
    AssertIndexRange(dimension, dim);
    return properties[SphericalParticle<dim, number>::Properties::force + dimension];
  }

  template <int dim, typename number>
  dealii::Tensor<1, dim, number>
  DEMParticleAccessor<dim, number>::force() const
  {
    dealii::Tensor<1, dim, number> force_vector;
    for (unsigned int dimension = 0; dimension < dim; ++dimension)
      force_vector[dimension] = force(dimension);
    return force_vector;
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
  number &
  DEMParticleAccessor<dim, number>::torque(const unsigned int dimension)
  {
    AssertIndexRange(dimension, axial_dim<dim>);
    return properties[SphericalParticle<dim, number>::Properties::torque + dimension];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::torque(const unsigned int dimension) const
  {
    AssertIndexRange(dimension, axial_dim<dim>);
    return properties[SphericalParticle<dim, number>::Properties::torque + dimension];
  }

  template <int dim, typename number>
  dealii::Tensor<1, axial_dim<dim>, number>
  DEMParticleAccessor<dim, number>::torque() const
  {
    dealii::Tensor<1, axial_dim<dim>, number> torque_vector;
    for (unsigned int dimension = 0; dimension < axial_dim<dim>; ++dimension)
      torque_vector[dimension] = torque(dimension);
    return torque_vector;
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
  dealii::types::particle_index
  DEMParticleAccessor<dim, number>::id() const
  {
    return particle_id;
  }

  template <int dim, typename number>
  dealii::types::particle_index
  DEMParticleAccessor<dim, number>::local_id() const
  {
    Assert(local_particle_id != dealii::numbers::invalid_unsigned_int,
           dealii::ExcMessage("The local particle ID is only valid for locally owned particles."));
    return local_particle_id;
  }

  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::radius()
  {
    return properties[SphericalParticle<dim, number>::Properties::radius];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::radius() const
  {
    return properties[SphericalParticle<dim, number>::Properties::radius];
  }


  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::mass()
  {
    return properties[SphericalParticle<dim, number>::Properties::mass];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::mass() const
  {
    return properties[SphericalParticle<dim, number>::Properties::mass];
  }

  template <int dim, typename number>
  number &
  DEMParticleAccessor<dim, number>::density()
  {
    return properties[SphericalParticle<dim, number>::Properties::density];
  }

  template <int dim, typename number>
  const number &
  DEMParticleAccessor<dim, number>::density() const
  {
    return properties[SphericalParticle<dim, number>::Properties::density];
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

  template <int dim, typename number>
  typename dealii::Triangulation<dim>::active_cell_iterator
  DEMParticleAccessor<dim, number>::get_surrounding_cell() const
  {
    Assert(surrounding_cell.state() == dealii::IteratorState::valid,
           dealii::ExcMessage("The surrounding cell is only valid for locally owned particles."));
    return surrounding_cell;
  }
} // namespace MeltPoolDG
