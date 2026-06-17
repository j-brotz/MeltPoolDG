#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/iterator_range.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/particles/particle_accessor.h>
#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/particles/dem_time_integrators.hpp>
#include <meltpooldg/particles/dem_util.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>
#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>

#include <ranges>
#include <type_traits>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  class ObstacleField
  {
  public:
    /**
     * @brief Constructor. Initializes the obstacle field and supporting data structures.
     *
     * This constructor reads obstacle input from file, initializes the internal particle handler,
     * and prepares the obstacle field for simulation. After construction, the obstacle field is
     * ready for further computations.
     *
     * @note Currently, only stationary obstacles are supported. A runtime assertion checks this
     * condition.
     *
     * @param data Obstacle-related configuration data.
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param mapping Mapping used to interpret geometry on the given triangulation.
     */
    ObstacleField(const ObstacleData<number>       &data,
                  const dealii::Triangulation<dim> &triangulation,
                  const dealii::Mapping<dim>       &mapping);

    /**
     * @brief Constructor. Initializes the obstacle field and supporting data structures.
     *
     * This constructor initializes the internal particle handler with the provided obstacle
     * locations and properties, preparing the obstacle field for simulation. After
     * construction, the obstacle field is ready for further computations.
     *
     * @param data Obstacle-related configuration data.
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param mapping Mapping used to interpret geometry on the given triangulation.
     * @param obstacle_locations Vector of obstacle center locations.
     * @param obstacle_properties Vector of obstacle properties corresponding to each location.
     */
    ObstacleField(const ObstacleData<number>                    &data,
                  const dealii::Triangulation<dim>              &triangulation,
                  const dealii::Mapping<dim>                    &mapping,
                  const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                  const std::vector<std::vector<number>>        &obstacle_properties);

    /**
     * @brief Advances the state of all obstacles in time by a single time step.
     *
     * @param current_time The current simulation time after the time step.
     * @param time_step The size of the time step to advance.
     * @param n_time_step The current time step number in the simulation.
     */
    void
    advance_time(const number current_time, const number time_step);

    /**
     * @brief Computes and applies the total load acting on each obstacle in the field. This
     * includes both forces and torques.
     *
     * This method iterates over all registered loads in the @p loads vector and computes the
     * cumulative load for each obstacle by summing the contributions of all individual load
     * models. The resulting total load (force and torque) is then stored in the obstacle’s
     * properties using the appropriate setter defined by @p ObstacleType.
     */
    void
    compute_loads_on_obstacles();

    /**
     * @brief Adds a new obstacle load to the list of loads acting on obstacles.
     *
     * This method appends the given load object to the internal list of load that are applied
     * when computing the total load on an obstacle -- this includes forces and torques.
     *
     * @param obstacle_load The load object to be added. It is forwarded and stored via type
     * erasure.
     */
    template <typename ObstacleLoadType>
      requires(!std::is_lvalue_reference_v<ObstacleLoadType>) //
    void
    add_load_type(ObstacleLoadType &&obstacle_load)
    {
      loads.emplace_back(std::move(obstacle_load));
    }

    /**
     * @brief Identifies obstacles that partially or fully occupy a given cell, and stores their
     * properties in the destination property pool.
     *
     * This function inspects the specified cell and collects the properties of any obstacles
     * located within it. These properties are stored in the provided destination property pool.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param cell The cell to be investigated.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const;

    /**
     * @brief Identify obstacles that partially or fully occupy any cell in a given cell batch, and
     * store their properties in the destination property pool.
     *
     * This function scans the specified cell batch and collects the properties of any obstacles
     * present in those cells. The properties are stored in the given destination property pool.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param matrix_free MatrixFree object associated with the current cell batch.
     * @param cell_batch_id Index of the cell batch to be examined.
     * @param n_lanes Number of vectorization lanes in the cell batch, i.e., the number of cells
     * present in the cell batch.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(
      dealii::Particles::PropertyPool<dim>                               &dst,
      const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const;


    /**
     * Computes the sum of all particle forces and prints the corresponding norm to the console.
     *
     * @note This function is intended to be used for testing purposes.
     */
    void
    print_accumulated_obstacle_force_norm(const dealii::ConditionalOStream pout) const;

    /**
     * @brief Performs the objects deserialization.
     *
     * This function forwards the call to dealii::ParticleHandler::deserialize(). See the
     * documentation of that function for further details.
     */
    void
    deserialize();

    /**
     * @brief Serializes the internal state of the class.
     *
     * This function forwards the call to dealii::ParticleHandler::serialize(). See the
     * documentation of that function for further details.
     *
     * @param ar       The archive used for serialization or deserialization.
     * @param version  The serialization version.
     */
    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int version);

    /**
     * @brief Prepares this object for serialization.
     *
     * This function forwards the call to dealii::ParticleHandler::prepare_for_serialization(). See
     * the documentation of that function for further details.
     */
    void
    prepare_for_serialization();

    /**
     * @brief Return the AMR regions relevant for the obstacle field in region-based refinement.
     *
     * This function computes and returns a vector of AMR regions corresponding to all particles
     * in the obstacle field. Each particle contributes a spherical shell region, with the shell
     * size determined by the parameters stored in the obstacle data structure associated with
     * the object's obstacle field.
     *
     * @return A vector of AMR regions for all particles in the field.
     *
     * @note The returned vector contains regions for all **global** particles, not just those
     * local to the current process or subdomain.
     */
    std::vector<AMR::AMRRegion<dim, number>>
    get_refinement_regions() const;

    /**
     * @brief Return a reference to the underlying particle handler of this object.
     *
     * @return The used particle handler of the current object.
     */
    dealii::Particles::ParticleHandler<dim> &
    get_particle_handler()
    {
      return obstacle_handler;
    }

    /**
     * @brief Return a constant reference to the underlying obstacle data structure of this object.
     *
     * @return The used obstacle data structure of the current object.
     */
    const ObstacleCompleteDomainSearch<dim, number, ObstacleType> &
    get_obstacle_data_structure() const
    {
      return obstacle_data_structure;
    }

    /**
     * @brief Insert obstacles into the particle handler based on provided locations and properties.
     *
     * This function takes a set of obstacle locations and their corresponding properties, and
     * inserts them into the internal particle handler. It ensures that the obstacles are properly
     * initialized and ready for simulation.
     *
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param obstacle_locations A vector of points representing the locations of the obstacles.
     * @param obstacle_properties A vector of vectors, where each inner vector contains the
     * properties associated with the corresponding obstacle location.
     */
    void
    insert_obstacles(const dealii::Triangulation<dim>              &triangulation,
                     const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                     const std::vector<std::vector<number>>        &obstacle_properties);

    /**
     * Returns a subrange for iterating over particles that are owned locally.
     *
     * @return An iterable subrange representing the local particles.
     */
    std::ranges::subrange<ParticleIterator<dim, number>>
    locally_owned_particle_range();

    /**
     * Returns a range over all global obstacle particles for iterating over all particles in the
     * global particle data structure.
     *
     * @return An iterable subrange representing all global particles.
     */
    std::ranges::subrange<ParticleIterator<dim, number>>
    global_particle_range();

  private:
    /// Struct holding configuration data for obstacles.
    const ObstacleData<number> &data;

    /// Vector of load objects representing all loads acting on the obstacles.
    std::vector<ObstacleLoad<dim, number, ObstacleType>> loads;

    /// Handler responsible for managing obstacle particles within the computational domain.
    dealii::Particles::ParticleHandler<dim> obstacle_handler;

    /// Obstacle search utility for locating relevant obstacles within a given cell or batch.
    /// TODO: Extend to support nearest-neighbor searches and other spatial queries.
    ObstacleCompleteDomainSearch<dim, number, ObstacleType> obstacle_data_structure;

    /// MPI communicator used for parallel operations on the obstacle field.
    MPI_Comm mpi_communicator;
  };

  template <int dim, typename number, typename ObstacleType>
  template <class Archive>
  void
  ObstacleField<dim, number, ObstacleType>::serialize(Archive &ar, const unsigned int version)
  {
    obstacle_handler.serialize(ar, version);
  }
} // namespace MeltPoolDG
