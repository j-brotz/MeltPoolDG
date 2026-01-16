#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include <vector>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  struct ObstacleDataStructureArborX
  {
  public:
    ObstacleDataStructureArborX(const dealii::Triangulation<dim> &triangulation,
                                const dealii::Mapping<dim>       &)
      : particle_handler_triangulation(triangulation.get_mpi_communicator())
      , particle_handler_mapping(1)
    {
      create_particle_handler_coarse_triangulation(triangulation);
      obstacle_handler = std::make_unique<dealii::Particles::ParticleHandler<dim>>(
        particle_handler_triangulation, particle_handler_mapping, ObstacleType::n_obstacle_properties);
    }

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
    reinit()
    {
      // To be implemented
    }

    dealii::Particles::ParticleIterator<dim>
    begin() const
    {
      return obstacle_handler->begin();
    }

    dealii::Particles::ParticleIterator<dim>
    end() const
    {
      return obstacle_handler->end();
    }

    dealii::Particles::ParticleIterator<dim>
    begin()
    {
      return obstacle_handler->begin();
    }

    dealii::Particles::ParticleIterator<dim>
    end()
    {
      return obstacle_handler->end();
    }

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
     *
     * @note While this function technically does the same as @p get_obstacles_in_cell_batch(), it
     * is significantly slower. Therefore whenever possible, prefer using the batch version.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const
    {
      // To be implemented
      return {};
    }

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
    get_obstacles_in_cell_batch(dealii::Particles::PropertyPool<dim>  &dst,
                                const dealii::MatrixFree<dim, number> &matrix_free,
                                const unsigned int                     cell_batch_id) const
    {
      // To be implemented
      return {};
    }

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
    broadcast_global_particles() const
    {
      // To be implemented
    }

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
    serialize(Archive &ar, const unsigned int version)
    {
      obstacle_handler->serialize(ar, version);
    }

    /**
     * @brief Prepares this object for serialization.
     *
     * This function forwards the call to dealii::ParticleHandler::prepare_for_serialization(). See
     * the documentation of that function for further details.
     */
    void
    prepare_for_serialization()
    {
      obstacle_handler->prepare_for_serialization();
    }

    /**
     * @brief Performs the objects deserialization.
     *
     * This function forwards the call to dealii::ParticleHandler::deserialize(). See the
     * documentation of that function for further details.
     */
    void
    deserialize()
    {
      // Assumes that triangulation.load() has already been called!
      obstacle_handler->deserialize();
    }

    /**
     * @brief Return a reference to the underlying particle handler of this object.
     *
     * @return The used particle handler of the current object.
     */
    dealii::Particles::ParticleHandler<dim> &
    get_particle_handler()
    {
      return *obstacle_handler;
    }

    const dealii::Particles::ParticleHandler<dim> &
    get_particle_handler() const
    {
      return *obstacle_handler;
    }

    void
    insert_obstacles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                     const std::vector<std::vector<number>>        &obstacle_properties)
    {
      std::vector<dealii::BoundingBox<dim>> local_bounding_box =
        dealii::GridTools::compute_mesh_predicate_bounding_box(
          particle_handler_triangulation, dealii::IteratorFilters::LocallyOwnedCell());
      std::vector<std::vector<dealii::BoundingBox<dim>>> global_bounding_box =
        dealii::Utilities::MPI::all_gather(mpi_communicator, local_bounding_box);

      obstacle_handler->insert_global_particles(
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ?
          obstacle_locations :
          std::vector<dealii::Point<dim, number>>{},
        global_bounding_box,
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ?
          obstacle_properties :
          std::vector<std::vector<number>>{});
    }

  private:
    ///
    dealii::parallel::shared::Triangulation<dim> particle_handler_triangulation;

    dealii::MappingQ<dim> particle_handler_mapping;

    /// Handler responsible for managing obstacle particles within the computational domain.
    std::unique_ptr<dealii::Particles::ParticleHandler<dim>> obstacle_handler;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    void
    create_particle_handler_coarse_triangulation(
      const dealii::Triangulation<dim> &reference_triangulation)
    {
      dealii::BoundingBox<dim> global_bounding_box =
        dealii::GridTools::compute_bounding_box(reference_triangulation);

      // In the case of curved domains the bounding boxes might slightly disagree on different
      // ranks. Therefore we take the biggest possible bounding box.
      auto [corner_1, corner_2] = global_bounding_box.get_boundary_points();

      if (dynamic_cast<const dealii::parallel::distributed::Triangulation<dim> *>(
            &reference_triangulation) != nullptr)
        {
          for (int i = 0; i < dim; ++i)
            {
              MPI_Allreduce(&corner_1[i], &corner_1[i], 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);
              MPI_Allreduce(&corner_2[i], &corner_2[i], 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
            }

          global_bounding_box = dealii::BoundingBox<dim>({corner_1, corner_2});
        }

      double min_radius =.1*global_bounding_box.side_length(0); // TODO

      // Create a triangulation which has ~square/cubic elements
      number min_vertex_distance = global_bounding_box.side_length(0);
      for (int i = 1; i < dim; ++i)
        min_vertex_distance = std::min(min_vertex_distance, global_bounding_box.side_length(i));

      std::vector<unsigned int> repetitions(dim, 1);
      for (int i = 0; i < dim; ++i)
        {
          repetitions[i] = std::ceil(global_bounding_box.side_length(i) / min_vertex_distance);
        }
      dealii::GridGenerator::subdivided_hyper_rectangle(particle_handler_triangulation,
                                                        repetitions,
                                                        corner_1,
                                                        corner_2);

      // refine until min vertex distance is less than min_radius
      unsigned int n_refinements = static_cast<unsigned int>(
        std::ceil(std::log2(min_vertex_distance / min_radius) / std::log2(0.5)));

        std::cout << "Refining particle handler triangulation " << n_refinements << " times."
                  << std::endl;
      particle_handler_triangulation.refine_global(n_refinements);
    }

    bool
    process_particle_in_cell(
      const typename dealii::Particles::PropertyPool<dim>::Handle        &src_handle,
      const dealii::CellAccessor<dim>                                    &cell,
      dealii::Particles::PropertyPool<dim>                               &dst,
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle> &target_handles) const
    {
      // To be implemented
      return false;
    }
  };
} // namespace MeltPoolDG