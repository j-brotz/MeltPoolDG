#pragma once

#include <deal.II/base/array_view.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>

#include <memory>
#include <vector>

namespace MeltPoolDG
{
  /**
   * @brief A simple search utility that performs a linear scan over all particles.
   *
   * This struct implements a very basic search algorithm by iterating over all particles
   * in the domain and individually checking whether each one satisfies the desired condition.
   *
   * @note This approach is not optimized for performance and is intended primarily for
   * debugging, prototyping, or use in small-scale simulations.
   */
  template <int dim, typename number, typename ObstacleType>
  struct ObstacleCompleteDomainSearch
  {
  public:
    /**
     * Constructor. Initializes the internal particle handler and property pool for managing
     * obstacles.
     *
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param mapping Mapping used to interpret geometry on the given triangulation.
     */
    explicit ObstacleCompleteDomainSearch(const dealii::Triangulation<dim> &triangulation,
                                          const dealii::Mapping<dim>       &mapping);

    /**
     * @brief Destructor. Explicitly deregisters all particles from the global obstacle property
     * pool.
     */
    ~ObstacleCompleteDomainSearch();

    /**
     * @brief Reinitializes the internal data structure by synchronizing obstacle data across all
     * MPI processes.
     *
     * This function communicates all locally owned obstacles to every other process in the MPI
     * communicator. As a result, each process obtains and stores a complete local copy of all
     * obstacles, regardless of ownership. This enables fully local access to obstacle data
     * during subsequent computations.
     */
    void
    reinit();

    /**
     * @brief Identify obstacles that partially or fully occupy the specified cell, and store their
     * properties in the destination property pool.
     *
     * This function scans a cell and collects the properties of all obstacles that intersect
     * with the cell. These properties are then stored in the provided destination property
     * pool.
     *
     * The identification strategy is based on a brute-force approach: the function iterates
     * over all globally known obstacles (previously synchronized), checks each one for
     * relevance to the specified cell, and includes it if applicable.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param cell The cell of interest.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const;

    /**
     * @brief Identify obstacles that partially or fully occupy any cell in the specified cell batch,
     * and store their properties in the destination property pool.
     *
     * This function scans a batch of cells (represented by a MatrixFree cell batch) and
     * collects the properties of all obstacles that intersect with any cell in the batch. These
     * properties are then stored in the provided destination property pool.
     *
     * The identification strategy is based on a brute-force approach: the function iterates
     * over all globally known obstacles (previously synchronized), checks each one for
     * relevance to the specified cell batch, and includes it if applicable.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param matrix_free MatrixFree object associated with the current cell batch.
     * @param cell_batch_id Index of the cell batch to be examined.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(
      dealii::Particles::PropertyPool<dim>                               &dst,
      const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const;

    /**
     * @brief Broadcasts obstacle properties of all locally owned particles to all MPI processes.
     *
     * This function ensures that each process has access to a complete copy of all obstacles,
     * regardless of ownership. It enables computations involving obstacles even on processes
     * that do not originally own them.
     *
     * Each process broadcasts its locally owned obstacles in turn, including both their
     * location and
     * associated properties. The data is stored in the @p properties_global_obstacles structure.
     */
    void
    broadcast_global_particles() const;

    /**
     * Prepares the obstacle data structure for coarsening and refinement of the underlying
     * triangulation. This call is necessary to ensure that the obstacle data structure remains
     * consistent with the mesh after coarsening and refinement operations.
     */
    void
    prepare_for_coarsening_and_refinement();

    /**
     * Unpacks the obstacle data structure after coarsening and refinement operations. This call is
     * necessary to ensure that the obstacle data structure remains consistent with the mesh after
     * coarsening and refinement operations.
     *
     * @note It is required that prepare_for_coarsening_and_refinement() has been called before the
     * triangulation was coarsened and refined.
     */
    void
    unpack_after_coarsening_and_refinement();

    /**
     * Insert obstacles into the particle data structure based on provided locations and properties.
     * This function takes a set of obstacle locations and their corresponding properties, and
     * inserts them into the internal particle handler. It ensures that the obstacles are properly
     * initialized and ready for simulation. Thereby it is not required that the particle locations
     * passed to this function are located in the local subdomain of the current MPI rank. The
     * function will automatically determine which MPI rank owns which particles and insert them
     * accordingly.
     *
     * @param obstacle_locations A vector of points representing the locations of the obstacles.
     * @param obstacle_properties A vector of vectors, where each inner vector contains the
     * properties associated with the corresponding obstacle location.
     */
    void
    insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                            const std::vector<std::vector<number>>        &obstacle_properties);

    /**
     * Registers the obstacle handler for particle output.
     *
     * @param postprocessor The postprocessor to which the particles are registered for output.
     */
    void
    register_particle_output(Postprocessor<dim, number> &postprocessor) const;

    /**
     * Returns a subrange for iterating over particles that are owned by the current MPI rank.
     *
     * @return An iterable subrange representing the local particles.
     */
    std::ranges::subrange<ParticleIterator<dim, number>>
    locally_owned_particle_range() const;

    /**
     * Returns a subrange for iterating over particles for which the particle center location is
     * contained in the specified active cell.
     *
     * @param cell The active cell for which to retrieve particles.
     * @return An iterable subrange representing the particles in the cell.
     */
    typename std::ranges::subrange<ParticleIterator<dim, number>>
    particles_in_cell(typename dealii::Triangulation<dim>::active_cell_iterator cell) const;

    /**
     * Prepares this object for serialization. This function forwards the call to
     * dealii::ParticleHandler::prepare_for_serialization(). See the documentation of that function
     * for further details.
     */
    void
    prepare_for_serialization();

    /**
     * Serializes the internal state of the class. This function forwards the call to
     * dealii::ParticleHandler::serialize(). See the documentation of that function for further
     * details.
     *
     * @param ar       The archive used for serialization or deserialization.
     * @param version  The serialization version.
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int version);

    /**
     * Performs the objects deserialization. This function forwards the call to
     * dealii::ParticleHandler::deserialize(). See the documentation of that function for further
     * details.
     */
    void
    deserialize();

    /**
     * Return the number of global particles, i.e., the total number of particles across all MPI
     * ranks.
     */
    unsigned int
    n_global_particles() const;

    /**
     * Return the number of locally owned particles, i.e., the number of particles owned by the
     * current MPI rank.
     */
    unsigned int
    n_locally_owned_particles() const;

    /**
     * Return a reference to a property pool containing the properties of all globally available
     * particles. The properties stored in the property pool represent those available in the field
     * when broadcast_global_particles() has been called the last time.
     */
    const dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties() const;

    /**
     * Same as above but returns a non-const reference to the property pool.
     */
    dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties();

  private:
    /// Handler managing the locally owned obstacles in the domain.
    dealii::Particles::ParticleHandler<dim> obstacle_handler;

    /// Property pool containing the properties of all global obstacles, stored locally on each
    /// MPI rank.
    mutable dealii::Particles::PropertyPool<dim> properties_global_obstacles;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    /**
     * @brief Deregisters all particles from the global obstacle property pool.
     */
    void
    deregister_property_pool() const;
  };

  template <int dim, typename number, typename ObstacleType>
  template <class Archive>
  void
  ObstacleCompleteDomainSearch<dim, number, ObstacleType>::serialize(Archive           &ar,
                                                                     const unsigned int version)
  {
    obstacle_handler.serialize(ar, version);
  }
} // namespace MeltPoolDG
