#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/particles/particle_accessor.h>
#include <deal.II/particles/particle_handler.h>

#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>

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
    ObstacleField(const ObstacleData               &data,
                  const dealii::Triangulation<dim> &triangulation,
                  const dealii::Mapping<dim>       &mapping);

    /**
     * @brief Computes and applies the total force acting on each obstacle in the field.
     *
     * This method iterates over all registered forces in the @p forces vector and computes the
     * cumulative force for each obstacle by summing the contributions of all individual force
     * models. The resulting total force is then stored in the obstacle’s properties using the
     * appropriate
     * setter defined by @p ObstacleType.
     */
    void
    compute_forces_on_obstacles();

    /**
     * @brief Adds a new obstacle force to the list of forces acting on obstacles.
     *
     * This method appends the given force object to the internal list of forces that are applied
     * when computing the total force on an obstacle.
     *
     * @param obstacle_force The force object to be added. It is forwarded and stored via type
     * erasure.
     */
    template <typename ObstacleForceType>
    void
    add_force_type(ObstacleForceType &&obstacle_force)
    {
      forces.push_back(ObstacleForce<dim, number, ObstacleType>(std::move(obstacle_force)));
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
    get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim>  &dst,
                                const dealii::MatrixFree<dim, number> &matrix_free,
                                const unsigned int                     cell_batch_id) const;


    /**
     * Computes the sum of all particle forces and prints the corresponding norm to the console.
     *
     * @note This function is intended to be used for testing purposes.
     */
    void
    print_accumulated_obstacle_force_norm(const dealii::ConditionalOStream pout) const;

    /**
     * @brief Rerturn a reference to the underlying particle handler of this object.
     *
     * @return The used particle handler of the current object.
     */
    dealii::Particles::ParticleHandler<dim> &
    get_particle_handler()
    {
      return obstacle_handler;
    }

    /**
     * @brief Rerturn a constant reference to the underlying obstacle data structure of this object.
     *
     * @return The used obstacle data structure of the current object.
     */
    const ObstacleCompleteDomainSearch<dim, number, ObstacleType> &
    get_obstacle_data_structure() const
    {
      return obstacle_data_structure;
    }

  private:
    /**
     * @brief Reads the obstacle state input file and initializes obstacles in the simulation domain.
     *
     * This function reads the obstacle data file specified in the configuration and generates
     * obstacle representations accordingly. It populates the internal particle handler with the
     * parsed obstacle positions and properties, preparing the obstacle field for use in subsequent
     * computations.
     *
     * @param triangulation The triangulation over which the obstacles are distributed.
     */
    void
    read_particle_state_input_file(const dealii::Triangulation<dim> &triangulation);

    /// Struct holding configuration data for obstacles.
    const ObstacleData &data;

    /// Vector of force objects representing all forces acting on the obstacles.
    std::vector<ObstacleForce<dim, number, ObstacleType>> forces;

    /// Handler responsible for managing obstacle particles within the computational domain.
    dealii::Particles::ParticleHandler<dim> obstacle_handler;

    /// Obstacle search utility for locating relevant obstacles within a given cell or batch.
    /// TODO: Extend to support nearest-neighbor searches and other spatial queries.
    ObstacleCompleteDomainSearch<dim, number, ObstacleType> obstacle_data_structure;

    /// MPI communicator used for parallel operations on the obstacle field.
    MPI_Comm mpi_communicator;
  };
} // namespace MeltPoolDG
