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
#include <meltpooldg/post_processing/postprocessor.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>

#include <ranges>
#include <type_traits>
#include <vector>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  class ObstacleField
  {
    using DataStructureType = ObstacleCompleteDomainSearch<dim, number, ObstacleType>;

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
    ObstacleField(const ObstacleData<number>            &data,
                  const dealii::Triangulation<dim>      &triangulation,
                  const dealii::Mapping<dim>            &mapping,
                  dealii::TimerOutput                   &timer,
                  const dealii::MatrixFree<dim, number> *matrix_free = nullptr);

    /**
     * @brief Constructor. Initializes the obstacle field and supporting data structures.
     *
     * This constructor initializes the internal particle handler with the provided
     * obstacle locations and properties, preparing the obstacle field for simulation.
     * After construction, the obstacle field is ready for further computations.
     *
     * @param data Obstacle-related configuration data.
     * @param triangulation The triangulation on which the obstacles are placed.
     * @param mapping Mapping used to interpret geometry on the given triangulation.
     * @param obstacle_locations Vector of obstacle center locations.
     * @param obstacle_properties Vector of obstacle properties corresponding to each location.
     */
    ObstacleField(const ObstacleData<number>              &data,
                  const dealii::Triangulation<dim>        &triangulation,
                  const dealii::Mapping<dim>              &mapping,
                  std::vector<dealii::Point<dim, number>> &obstacle_locations,
                  std::vector<std::vector<number>>        &obstacle_properties,
                  dealii::TimerOutput                     &timer,
                  const dealii::MatrixFree<dim, number>   *matrix_free = nullptr);

    /**
     * @brief Advances the state of all obstacles in time by a single time step.
     *
     * @param current_time The current simulation time after the time step.
     * @param time_step The size of the time step to advance.
     * @param n_time_step The current time step number in the simulation.
     */
    void
    advance_time(const number                      current_time,
                 const number                      time_step,
                 const dealii::ConditionalOStream &pout);

    /**
     * @brief Computes and applies the total load acting on each obstacle in the field. This
     * includes both forces and torques.
     *
     * This method iterates over all registered loads in the @p loads vector and computes the
     * cumulative load for each obstacle by summing the contributions of all individual
     * load models. The resulting total load (force and torque) is then stored in the
     * obstacle’s
     * properties using the appropriate setter defined by @p ObstacleType.
     */
    void
    compute_loads_on_obstacles();

    /**
     * @brief Adds a new obstacle load to the list of loads acting on obstacles.
     *
     * This method appends the given load object to the internal list of load that are
     * applied when computing the total load on an obstacle -- this includes forces and
     * torques.
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
     * @brief Identify obstacles that partially or fully occupy any cell in a given cell batch, and
     * store their properties in the destination property pool.
     *
     * This function scans the specified cell batch and collects the properties of any
     * obstacles present in those cells. The properties are stored in the given
     * destination property pool.
     *
     * @param cells Vector of cells to be examined.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    void
    get_obstacles_in_cell(
      const boost::container::small_vector_base<dealii::TriaIterator<dealii::CellAccessor<dim>>>
                                                                            &cells,
      boost::container::small_vector_base<DEMParticleAccessor<dim, number>> &obstacles) const
    {
      obstacle_data_structure.get_obstacles_in_cell(cells, obstacles);
    }

    void
    get_obstacles_in_cell(
      const dealii::TriaIterator<dealii::CellAccessor<dim>>                 &cell,
      boost::container::small_vector_base<DEMParticleAccessor<dim, number>> &obstacles) const
    {
      obstacle_data_structure.get_obstacles_in_cell(cell, obstacles);
    }

    std::vector<MeltPoolDG::DEMParticleAccessor<dim, number>> &
    get_obstacles_in_cell_batch(const unsigned int cell_batch_id) const
    {
      return obstacle_data_structure.get_obstacles_in_cell_batch(cell_batch_id);
    }

    void
    register_particle_output(Postprocessor<dim, number> &postprocessor)
    {
      obstacle_data_structure.register_particle_output(postprocessor);
    }

    /**
     * Computes the sum of all particle forces and prints the corresponding norm to the
     * console.
     *
     * @note This function is intended to be used for testing purposes.
     */
    void
    print_accumulated_obstacle_force_norm(const dealii::ConditionalOStream pout);

    void
    prepare_for_coarsening_and_refinement()
    {
      obstacle_data_structure.prepare_for_coarsening_and_refinement();
    }

    void
    unpack_after_coarsening_and_refinement()
    {
      obstacle_data_structure.unpack_after_coarsening_and_refinement();
      dynamic_update_control.reinit_after_update();
    }


    /**
     * @brief Performs the objects deserialization.
     *
     * This function forwards the call to dealii::ParticleHandler::deserialize(). See
     * the documentation of that function for further details.
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
     * This function forwards the call to
     * dealii::ParticleHandler::prepare_for_serialization(). See the documentation of
     * that function for further details.
     */
    void
    prepare_for_serialization();

    /**
     * @brief Return the AMR regions relevant for the obstacle field in region-based refinement.
     *
     * This function computes and returns a vector of AMR regions corresponding to all
     * particles in the obstacle field. Each particle contributes a spherical shell
     * region, with the shell size determined by the parameters stored in the obstacle
     * data structure associated with the object's obstacle field.
     *
     * @return A vector of AMR regions for all particles in the field.
     *
     * @note The returned vector contains regions for all **global** particles, not just those
     * local to the current process or subdomain.
     */
    std::vector<AMR::AMRRegion<dim, number>>
    get_refinement_regions();

    /**
     * @brief Insert obstacles into the particle handler based on provided locations and properties.
     *
     * This function takes a set of obstacle locations and their corresponding
     * properties, and inserts them into the internal particle handler. It ensures that
     * the obstacles are properly initialized and ready for simulation.
     *
     * @param obstacle_locations A vector of points representing the locations of the obstacles.
     * @param obstacle_properties A vector of vectors, where each inner vector contains the
     * properties associated with the corresponding obstacle location.
     */
    void
    insert_obstacles(std::vector<dealii::Point<dim, number>> &obstacle_locations,
                     std::vector<std::vector<number>>        &obstacle_properties);

    /**
     * Returns a subrange for iterating over particles that are owned locally.
     *
     * @return An iterable subrange representing the local particles.
     */
    std::ranges::subrange<ParticleIterator<dim, number>>
    locally_owned_particle_range();

    std::ranges::subrange<ParticleIterator<dim, number>>
    ghost_particle_range();

    void
    compress()
    {
      obstacle_data_structure.compress();
    }

    /**
     *
     */
    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle,
                      const number                            relative_tolerance) const
    {
      return obstacle_data_structure.contact_particles(particle, relative_tolerance);
    }

  private:
    /**
     * @brief Reads the obstacle state input file and returns obstacle locations and properties.
     *
     * This function reads the obstacle data file specified in the configuration and
     * generates obstacle representations accordingly. The resulting vectors for the
     * obstacle properties and locations are returned. It populates the internal
     * particle handler with the parsed obstacle positions and properties, preparing the
     * obstacle field for use in subsequent computations.
     *
     * @return A pair consisting of a vector of obstacle locations and a vector of  corresponding
     * properties.
     */
    std::pair<std::vector<dealii::Point<dim, number>>, std::vector<std::vector<number>>>
    read_obstacle_state_input_file();

    /// Struct holding configuration data for obstacles.
    const ObstacleData<number> &data;

    /// Vector of load objects representing all loads acting on the obstacles.
    std::vector<ObstacleLoad<dim, number, ObstacleType>> loads;

    /// Obstacle search utility for locating relevant obstacles within a given cell or
    /// batch.
    ObstacleCompleteDomainSearch<dim, number, ObstacleType> obstacle_data_structure;

    /// MPI communicator used for parallel operations on the obstacle field.
    MPI_Comm mpi_communicator;

    /// Timer data for profiling the obstacle field operations.
    dealii::TimerOutput &timer;

    DynamicUpdateController<dim, number> dynamic_update_control;
  };

  // TODO
  template <int dim, typename number, typename ObstacleType>
  template <class Archive>
  void
  ObstacleField<dim, number, ObstacleType>::serialize(Archive &ar, const unsigned int version)
  {
    // obstacle_handler.serialize(ar, version);
  }
} // namespace MeltPoolDG
