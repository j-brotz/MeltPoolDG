#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/property_pool.h>

#include "meltpooldg/particles/particle_accessor.hpp"
#include <meltpooldg/particles/particle_iterator.hpp>
#include <meltpooldg/post_processing/postprocessor.hpp>

#include <boost/container/small_vector.hpp>

#include <cmath>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

#include "mpi.h"

namespace MeltPoolDG
{
  enum class ObstacleDataStructureType
  {
    CompleteDomainSearch,
    CellBasedSearch
  };



  /**
   * @brief Interface class for obstacle data structures supporting efficient spatial queries,
   * such as nearest-neighbor searches.
   *
   * This interface abstracts different obstacle search algorithms using type erasure, allowing
   * flexible integration of various implementations. Any obstacle search algorithm must conform
   * to this interface to be used within the framework.
   *
   * The following member functions must be implemented:
   * - @p reinit(): Reinitializes the internal state. This should be called after obstacle positions
   *   have changed or the underlying triangulation has been modified.
   * - @p get_obstacles_in_cell(): Identifies all obstacles that partially or fully occupy a given
   *   cell and stores their properties in a specified destination.
   * - @p get_obstacles_in_cell_batch(): Identifies all obstacles that intersect with any cell in a
   *   given cell batch and stores their properties in a specified destination.
   *
   * See below for more details.
   */
  template <int dim, typename number>
  struct ObstacleDataStructure
  {
  public:
    /**
     * Constructor. Store the passed obstacle data structure internally.
     *
     * @param obstacle_data_structure Concrete obstacle data structure used in the class.
     */
    template <typename ObstacleDataStructureType>
    explicit ObstacleDataStructure(ObstacleDataStructureType &&obstacle_data_structure_in)
      : obstacle_data_structure_pimpl(
          std::make_unique<ObstacleDataStructureModel<ObstacleDataStructureType>>(
            std::move(obstacle_data_structure_in)))
    {}

    /**
     * @brief Reinitializes the internal data structure.
     */
    void
    reinit()
    {
      obstacle_data_structure_pimpl->reinit();
    }

    /**
     * @brief Identify obstacles that partially or fully occupy the specified cell, and store their
     * properties in the destination property pool.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param cell The cell of interest.
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                          const dealii::CellAccessor<dim>      &cell) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(dst, cell);
    }


    /**
     * @brief Identify obstacles that partially or fully occupy any cell in the specified cell batch,
     * and store their properties in the destination property pool.
     *
     * @param dst Destination property pool where the properties of the identified obstacles will be
     * stored.
     * @param matrix_free MatrixFree object associated with the current cell batch.
     * @param cell_batch_id Index of the cell batch to be examined.
     * @param n_lanes Number of vectorization lanes in the cell batch (i.e., the number of cells in
     * the batch).
     *
     * @return Vector containing the handles of the newly registered obstacles in @p dst.
     */
    std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
    get_obstacles_in_cell(
      dealii::Particles::PropertyPool<dim>                               &dst,
      const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(dst, cells);
    }

    const dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties() const
    {
      return obstacle_data_structure_pimpl->get_global_particle_properties();
    }

    dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties()
    {
      return obstacle_data_structure_pimpl->get_global_particle_properties();
    }

    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle,
                      const number                            relative_tolerance) const
    {
      return obstacle_data_structure_pimpl->contact_particles(particle, relative_tolerance);
    }

    void
    prepare_for_serialization()
    {
      obstacle_data_structure_pimpl->prepare_for_serialization();
    }

    void
    deserialize()
    {
      obstacle_data_structure_pimpl->deserialize();
    }

    unsigned int
    n_global_particles() const
    {
      return obstacle_data_structure_pimpl->n_global_particles();
    }

    unsigned int
    n_locally_owned_particles() const
    {
      return obstacle_data_structure_pimpl->n_locally_owned_particles();
    }

    std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
    locally_owned_particle_range()
    {
      return obstacle_data_structure_pimpl->locally_owned_particle_range();
    }

    void
    insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                            const std::vector<std::vector<number>>        &obstacle_properties)
    {
      obstacle_data_structure_pimpl->insert_global_particles(obstacle_locations,
                                                             obstacle_properties);
    }

    void
    update_ghost_particle_properties()
    {
      obstacle_data_structure_pimpl->update_ghost_particle_properties();
    }

    void
    sort_particles_into_subdomains_and_cells()
    {
      obstacle_data_structure_pimpl->sort_particles_into_subdomains_and_cells();
    }

    std::vector<DEMParticleAccessor<dim, number>>
    get_obstacles_in_cell(const dealii::CellAccessor<dim> &cell)
    {
      return obstacle_data_structure_pimpl->get_obstacles_in_cell(cell);
    }

    void
    register_particle_output(Postprocessor<dim, number> &postprocessor)
    {
      obstacle_data_structure_pimpl->register_particle_output(postprocessor);
    }

    void
    prepare_for_coarsening_and_refinement()
    {
      obstacle_data_structure_pimpl->prepare_for_coarsening_and_refinement();
    }

    void
    unpack_after_coarsening_and_refinement()
    {
      obstacle_data_structure_pimpl->unpack_after_coarsening_and_refinement();
    }

  private:
    /**
     * @brief Concept: Abstract interface for obstacle data structures.
     *
     * Implementations must define how to reinitialize the structure, and  how to extract obstacles
     * from single cells or batches.See public interface for detailed function descriptions.
     */
    struct ObstacleDataStructureConcept
    {
      virtual ~ObstacleDataStructureConcept() = default;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      virtual void
      reinit() = 0;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      virtual std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                            const dealii::CellAccessor<dim>      &cell) const = 0;

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      virtual std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(
        dealii::Particles::PropertyPool<dim>                               &dst,
        const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const = 0;

      virtual const dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() const = 0;

      virtual dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() = 0;

      virtual boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
      contact_particles(const DEMParticleAccessor<dim, number> &particle,
                        const number                            relative_tolerance) const = 0;

      virtual void
      prepare_for_serialization() = 0;

      virtual void
      deserialize() = 0;

      virtual unsigned int
      n_global_particles() const = 0;

      virtual unsigned int
      n_locally_owned_particles() const = 0;

      virtual std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
      locally_owned_particle_range() = 0;

      virtual void
      insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                              const std::vector<std::vector<number>> &obstacle_properties) = 0;

      virtual void
      update_ghost_particle_properties() = 0;

      virtual void
      sort_particles_into_subdomains_and_cells() = 0;

      virtual std::vector<DEMParticleAccessor<dim, number>>
      get_obstacles_in_cell(const dealii::CellAccessor<dim> &cell) = 0;

      virtual void
      register_particle_output(Postprocessor<dim, number> &postprocessor) = 0;

      virtual void
      prepare_for_coarsening_and_refinement() = 0;

      virtual void
      unpack_after_coarsening_and_refinement() = 0;
    };

    /**
     * @brief Model required for type erasure. See public interface for detailed function
     * descriptions.
     */
    template <typename ObstacleDataStructureType>
    struct ObstacleDataStructureModel final : public ObstacleDataStructureConcept
    {
      /**
       * Constructor. Stores the given obstacle data structure internally.
       *
       * @param obstacle_data_structure Data structure of the obstacles.
       */
      explicit ObstacleDataStructureModel(ObstacleDataStructureType &&obstacle_data_structure)
        : obstacle_data_structure(std::move(obstacle_data_structure))
      {}

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      void
      reinit() override
      {
        obstacle_data_structure.reinit();
      }

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(dealii::Particles::PropertyPool<dim> &dst,
                            const dealii::CellAccessor<dim>      &cell) const override
      {
        return obstacle_data_structure.get_obstacles_in_cell(dst, cell);
      }

      /**
       * Part of the type erasure interface. Refer to the public interface documentation for more
       * details.
       */
      std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>
      get_obstacles_in_cell(
        dealii::Particles::PropertyPool<dim>                               &dst,
        const std::vector<dealii::TriaIterator<dealii::CellAccessor<dim>>> &cells) const override
      {
        return obstacle_data_structure.get_obstacles_in_cell(dst, cells);
      }

      const dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() const override
      {
        return obstacle_data_structure.get_global_particle_properties();
      }

      dealii::Particles::PropertyPool<dim> &
      get_global_particle_properties() override
      {
        return obstacle_data_structure.get_global_particle_properties();
      }

      boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
      contact_particles(const DEMParticleAccessor<dim, number> &particle,
                        const number                            relative_tolerance) const override
      {
        return obstacle_data_structure.contact_particles(particle, relative_tolerance);
      }

      void
      prepare_for_serialization() override
      {
        obstacle_data_structure.prepare_for_serialization();
      }

      void
      deserialize() override
      {
        obstacle_data_structure.deserialize();
      }

      unsigned int
      n_global_particles() const override
      {
        return obstacle_data_structure.n_global_particles();
      }

      unsigned int
      n_locally_owned_particles() const override
      {
        return obstacle_data_structure.n_locally_owned_particles();
      }

      std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
      locally_owned_particle_range() override
      {
        return obstacle_data_structure.locally_owned_particle_range();
      }

      void
      insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                              const std::vector<std::vector<number>> &obstacle_properties) override
      {
        obstacle_data_structure.insert_global_particles(obstacle_locations, obstacle_properties);
      }

      void
      update_ghost_particle_properties() override
      {
        obstacle_data_structure.update_ghost_particle_properties();
      }

      void
      sort_particles_into_subdomains_and_cells() override
      {
        obstacle_data_structure.sort_particles_into_subdomains_and_cells();
      }

      std::vector<DEMParticleAccessor<dim, number>>
      get_obstacles_in_cell(const dealii::CellAccessor<dim> &cell) override
      {
        return obstacle_data_structure.get_obstacles_in_cell(cell);
      }

      void
      register_particle_output(Postprocessor<dim, number> &postprocessor) override
      {
        obstacle_data_structure.register_particle_output(postprocessor);
      }

      void
      prepare_for_coarsening_and_refinement() override
      {
        obstacle_data_structure.prepare_for_coarsening_and_refinement();
      }

      void
      unpack_after_coarsening_and_refinement() override
      {
        obstacle_data_structure.unpack_after_coarsening_and_refinement();
      }


    private:
      ObstacleDataStructureType obstacle_data_structure;
    };

    /// Pointer to the concrete obstacle data structure used within this class.
    std::unique_ptr<ObstacleDataStructureConcept> obstacle_data_structure_pimpl;
  };

  // TODO: Find a better name for this class
  template <int dim>
  class LevelCellPartitioner
  {
  public:
    LevelCellPartitioner(const dealii::Triangulation<dim> &tria, const unsigned int level)
      : triangulation(tria)
      , partition_level(level)
      , mpi_communicator(triangulation.get_mpi_communicator())
    {}

    void
    reinit()
    {
      setup_communication_pattern();
    }

    /**
     * Return a map that assigns to each MPI rank the cell ids of the locally owned cells for which
     * the current process needs to send data to the corresponding other rank.
     */
    std::map<int, std::vector<dealii::CellId>>
    get_cell_to_rank_send() const
    {
      return cell_to_rank_send;
    }

    /**
     * Return the number of processes to which the current process needs to send data.
     */
    unsigned
    n_processes_to_send_to() const
    {
      return cell_to_rank_send.size();
    }

    /**
     * Return the number of processes from which the current process needs to receive data.
     */
    unsigned
    n_processes_to_receive_from() const
    {
      return cell_to_rank_receive.size();
    }

    std::map<int, std::vector<dealii::CellId>>
    get_particle_receiver_ranks_and_cells()
    {
      return cell_to_rank_receive;
    }

    /**
     * Return a vector of MPI ranks from which the current process needs to receive data.
     */
    std::vector<int>
    get_particle_receiver_ranks()
    {
      // TODO: Make this more efficient
      std::vector<int> received_ranks;
      received_ranks.reserve(cell_to_rank_receive.size());
      for (const auto &rank : std::views::keys(cell_to_rank_receive))
        {
          received_ranks.push_back(rank);
        }
      return received_ranks;
    }

    std::vector<int>
    get_particle_sender_ranks()
    {
      // TODO: Make this more efficient
      std::vector<int> sender_ranks;
      sender_ranks.reserve(cell_to_rank_send.size());
      for (const auto &rank : std::views::keys(cell_to_rank_send))
        {
          sender_ranks.push_back(rank);
        }
      return sender_ranks;
    }

  private:
    /**
     * This function returns a set of parent cells on the partition level @p partition_level which
     * have at least one locally owned active cell as descendant. Hereby it is not checked whether
     * the parent cell is locally owned a ghost cell or an artificial cell.
     *
     * @return Vector of cell ids of the parent cells on the partition level.
     *
     * @note This function assumes that the relevant parent cells on the partition level are at
     * least available as artificial cells on the MPI process. This is guaranteed if the
     * triangulation has been created with the multigrid hierarchy setting turned on.
     */
    std::vector<dealii::CellId>
    level_parent_cells_of_owned_active_cells()
    {
      // lambda that determines for a given cell and a given level the parent cell on that level
      const auto level_parent_cell_id = [](const auto                       level_parent_cell_id,
                                           const dealii::CellAccessor<dim> &cell,
                                           int level) -> dealii::CellId {
        if (cell.level() == level)
          return cell.id();
        else
          return level_parent_cell_id(level_parent_cell_id, *cell.parent(), level);
      };

      std::set<dealii::CellId> parent_cells;
      for (auto &cell : triangulation.active_cell_iterators())
        {
          parent_cells.insert(level_parent_cell_id(level_parent_cell_id, *cell, partition_level));
        }

      return {std::vector<dealii::CellId>(parent_cells.begin(), parent_cells.end())};
    }

    /**
     * This function performs an all gather operation such that each MPI process receives the cell
     * ids of the cells on the partition level relevant for all other processes.
     *
     * @param local_parent_cells Vector of cell ids of the parent cells on the partition level which
     * are relevant for the local MPI process.
     *
     * @return Vector of pairs of MPI rank and vector of cell ids of the relevant cells on the
     * partition level for the corresponding MPI process.
     */
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
    all_gather_parent_cells(const std::vector<dealii::CellId> &local_parent_cells)
    {
      std::vector<std::pair<unsigned int, std::vector<dealii::CellId>>> all_parent_cells =
        dealii::Utilities::MPI::all_gather(mpi_communicator,
                                           std::make_pair(dealii::Utilities::MPI::this_mpi_process(
                                                            mpi_communicator),
                                                          local_parent_cells));

      return all_parent_cells;
    }

    /**
     * For a specific cell with given cell id find all adjacent cells on the same level which are
     * locally owned. This includes the cell itself if it is locally owned. We consider two cells to
     * be adjacent if they share at least a vertex.
     *
     * @param cell_id Cell id of the cell for which the adjacent cells should be found.
     *
     * @return Vector of cell ids of the adjacent cells which are locally owned.
     */
    std::vector<dealii::CellId>
    locally_relevant_cells_for_cell(const dealii::CellId &cell_id)
    {
      std::vector<dealii::CellId> relevant_cells;

      typename dealii::Triangulation<dim>::cell_iterator cell =
        triangulation.create_cell_iterator(cell_id);

      // If the cell is locally owned it is relevant
      if (cell->is_locally_owned_on_level())
        {
          relevant_cells.push_back(cell_id);
        }

      auto search_neighbors =
        [&relevant_cells](const auto &search_neighbors,
                          const typename dealii::Triangulation<dim>::cell_iterator &cell,
                          std::vector<unsigned> excluded_indices = {},
                          int                   current_depth    = 1) -> void {
        for (unsigned int neighbor_index = 0;
             neighbor_index < dealii::GeometryInfo<dim>::faces_per_cell;
             ++neighbor_index)
          {
            if (std::ranges::find(excluded_indices, neighbor_index) == excluded_indices.end() and
                not cell->at_boundary(neighbor_index) and
                cell->neighbor(neighbor_index)->is_locally_owned_on_level())
              {
                relevant_cells.push_back(cell->neighbor(neighbor_index)->id());
                if (current_depth < dim)
                  {
                    excluded_indices.push_back(neighbor_index);
                    search_neighbors(search_neighbors,
                                     cell->neighbor(neighbor_index),
                                     excluded_indices,
                                     current_depth + 1);
                  }
              }
          }
      };

      search_neighbors(search_neighbors, cell);

      std::sort(relevant_cells.begin(), relevant_cells.end());
      relevant_cells.erase(std::unique(relevant_cells.begin(), relevant_cells.end()),
                           relevant_cells.end());

      return relevant_cells;
    }

    /**
     * For each MPI process, find the locally owned relevant cells for the cells of interest.
     *
     * @param other_ranks_cell_of_interests Vector of pairs of MPI rank and vector of cell ids of
     * the relevant cells on the partition level for the corresponding MPI process.
     *
     * @return A vector of pairs with the first element being the MPI rank and the second element
     * being a vector of cell ids of the adjacent locally owned cells which are relevant for the
     * corresponding MPI process.
     */
    std::vector<std::pair<int, std::vector<dealii::CellId>>>
    locally_owned_relevant_cells_for_ranks(
      const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
        &other_ranks_cell_of_interests)
    {
      std::vector<std::pair<int, std::vector<dealii::CellId>>>
        locally_owned_relevant_cells_for_ranks;

      for (const auto &[rank, cells] : other_ranks_cell_of_interests)
        {
          if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
            {
              std::set<dealii::CellId> locally_owned_relevant_cells_for_rank;
              for (const auto &cell_id : cells)
                {
                  std::vector<dealii::CellId> relevant_cells =
                    locally_relevant_cells_for_cell(cell_id);
                  locally_owned_relevant_cells_for_rank.insert(relevant_cells.begin(),
                                                               relevant_cells.end());
                }
              locally_owned_relevant_cells_for_ranks.emplace_back(
                rank,
                std::vector<dealii::CellId>(locally_owned_relevant_cells_for_rank.begin(),
                                            locally_owned_relevant_cells_for_rank.end()));
            }
        }

      return locally_owned_relevant_cells_for_ranks;
    }

    /**
     *
     *
     * @param locally_owned_relevant_cells_for_ranks A vector of pairs with the first element being
     * the MPI rank and the second element being the corresponding relevant cells for which the
     * corresponding MPI process will receive data from the current process.
     *
     * @return A vector of pairs with the first element being the MPI rank and the second element
     * being the corresponding relevant cells for which the current MPI process will receive data
     * from the corresponding MPI process.
     */
    std::vector<std::pair<int, std::vector<dealii::CellId>>>
    send_and_receive_relevant_locally_owned_cells(
      const std::vector<std::pair<int, std::vector<dealii::CellId>>>
        &locally_owned_relevant_cells_for_ranks)
    {
      // send
      std::vector<dealii::Utilities::MPI::Future<void>> send_futures;
      send_futures.reserve(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) - 1);
      for (const auto &[rank, cells] : locally_owned_relevant_cells_for_ranks)
        {
          if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
            {
              send_futures.push_back(dealii::Utilities::MPI::isend(
                std::make_pair(dealii::Utilities::MPI::this_mpi_process(mpi_communicator), cells),
                mpi_communicator,
                rank,
                0));
            }
        }

      // receive
      std::vector<dealii::Utilities::MPI::Future<std::pair<int, std::vector<dealii::CellId>>>>
        receive_futures;
      for (unsigned int rank = 0; rank < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++rank)
        {
          if (rank != dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
            {
              receive_futures.push_back(
                dealii::Utilities::MPI::irecv<std::pair<int, std::vector<dealii::CellId>>>(
                  mpi_communicator, rank, 0));
            }
        }


      // wait
      std::vector<std::pair<int, std::vector<dealii::CellId>>> received_data;
      received_data.reserve(receive_futures.size());
      for (auto &future : receive_futures)
        {
          future.wait();
          received_data.push_back(future.get());
        }
      for (auto &future : send_futures)
        future.wait();

      return received_data;
    }



    /**
     * Stores the communication pattern for later use.
     */
    void
    store_communication_pattern(
      const std::vector<std::pair<int, std::vector<dealii::CellId>>> &data_to_receive,
      const std::vector<std::pair<int, std::vector<dealii::CellId>>> &data_to_send)
    {
      for (const auto &[rank, cells] : data_to_receive)
        {
          if (not cells.empty())
            cell_to_rank_receive[rank].insert(cell_to_rank_receive[rank].end(),
                                              cells.begin(),
                                              cells.end());
        }

      for (const auto &[rank, cells] : data_to_send)
        {
          if (not cells.empty())
            {
              cell_to_rank_send[rank].insert(cell_to_rank_send[rank].end(),
                                             cells.begin(),
                                             cells.end());
            }
        }
    }

    void
    setup_communication_pattern()
    {
      cell_to_rank_send.clear();
      cell_to_rank_receive.clear();

      // parent cells which have at least one locally owned active cell as descendant
      std::vector<dealii::CellId> locally_relevant_parent_cells =
        level_parent_cells_of_owned_active_cells();

      // broadcast those parent cells to all ranks
      std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> all_parent_cells =
        all_gather_parent_cells(locally_relevant_parent_cells);

      // find locally owned cells which are relevant for the ranks which sent parent cells of
      // interest
      std::vector<std::pair<int, std::vector<dealii::CellId>>> local_relevant_cells_for_ranks =
        locally_owned_relevant_cells_for_ranks(all_parent_cells);

      // inform other processes which own cells are relevant for them and get the information which
      // cells are relevant for us
      std::vector<std::pair<int, std::vector<dealii::CellId>>> received_data =
        send_and_receive_relevant_locally_owned_cells(local_relevant_cells_for_ranks);

      // Based on the selected cells to send and receive, store the communication pattern for later
      // use
      store_communication_pattern(received_data, local_relevant_cells_for_ranks);
    }

    std::map<int, std::vector<dealii::CellId>> cell_to_rank_send;

    std::map<int, std::vector<dealii::CellId>> cell_to_rank_receive;

    const dealii::Triangulation<dim> &triangulation;

    unsigned int partition_level;

    const MPI_Comm mpi_communicator;
  };


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
    ObstacleCompleteDomainSearch(const dealii::Triangulation<dim> &triangulation,
                                 const dealii::Mapping<dim>       &mapping,
                                 dealii::TimerOutput              &timer);

    /**
     * @brief Destructor. Explicitly deregisters all particles from the global obstacle property
     * pool.
     */
    ~ObstacleCompleteDomainSearch();

    ObstacleCompleteDomainSearch(ObstacleCompleteDomainSearch &&) = default;

    ObstacleCompleteDomainSearch &
    operator=(ObstacleCompleteDomainSearch &&) = default;

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

    std::vector<DEMParticleAccessor<dim, number>>
    get_obstacles_in_cell(const dealii::CellAccessor<dim> &cell)
    {
      dealii::TimerOutput::Scope t(timer, "particles in cell search");

      std::vector<DEMParticleAccessor<dim, number>> particles_in_cell;

      return particles_in_cell;
    }

    /**
     * Return a reference to a property pool containing the properties of all globally available
     * particles. The properties stored in the property pool represent those available in the
     * field when broadcast_global_particles() has been called the last time.
     */
    const dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties() const
    {
      return *properties_global_obstacles;
    }

    dealii::Particles::PropertyPool<dim> &
    get_global_particle_properties()
    {
      return *properties_global_obstacles;
    }

    boost::container::small_vector<DEMParticleAccessor<dim, number>, 3 * dim>
    contact_particles(const DEMParticleAccessor<dim, number> &particle,
                      const number                            relative_tolerance) const;

    void
    prepare_for_serialization()
    {
      obstacle_handler->prepare_for_serialization();
    }

    void
    deserialize()
    {
      obstacle_handler->deserialize();
    }

    unsigned int
    n_global_particles() const
    {
      return obstacle_handler->n_global_particles();
    }

    unsigned int
    n_locally_owned_particles() const
    {
      return obstacle_handler->n_locally_owned_particles();
    }

    std::ranges::subrange<MeltPoolDG::ParticleIterator<dim, number>>
    locally_owned_particle_range()
    {
      return std::ranges::subrange<ParticleIterator<dim, number>>(
        ParticleIterator<dim, number>(obstacle_handler->begin()),
        ParticleIterator<dim, number>(obstacle_handler->end()));
    }



    void
    insert_global_particles(const std::vector<dealii::Point<dim, number>> &obstacle_locations,
                            const std::vector<std::vector<number>>        &obstacle_properties)
    {
      // Update the maximum particle radius based on the received particles. As this function is a
      // collective operation, we can be sure that all processes receive the same particles and thus
      // have the same maximum radius after this step. There is no need for an additional global
      // reduction operation.
      for (const auto &properties : obstacle_properties)
        {
          max_particle_radius =
            std::max(max_particle_radius, properties[ObstacleType::Properties::radius]);
        }

      std::vector<dealii::BoundingBox<dim>> local_bounding_box =
        dealii::GridTools::compute_mesh_predicate_bounding_box(
          obstacle_handler->get_triangulation(), dealii::IteratorFilters::LocallyOwnedCell());
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

      // reinit();
    }

    void
    update_ghost_particle_properties()
    {
      // TODO
      dealii::TimerOutput::Scope t(timer, "update ghost particles");
      broadcast_global_particles();
    }

    void
    sort_particles_into_subdomains_and_cells()
    {
      dealii::TimerOutput::Scope t(timer, "update ghost particles");
      broadcast_global_particles();

      sort_particles_into_local_level_cells();
      communicate_ghost_particles();
    }

    void
    register_particle_output(Postprocessor<dim, number> &postprocessor)
    {
      const auto [property_names, property_component_interpretations] =
        ObstacleType::get_property_names_and_component_interpretation();

      postprocessor.register_obstacle_output(obstacle_handler.get(),
                                             property_names,
                                             property_component_interpretations);
    }

    void
    prepare_for_coarsening_and_refinement()
    {
      obstacle_handler->prepare_for_coarsening_and_refinement();
    }

    void
    unpack_after_coarsening_and_refinement()
    {
      obstacle_handler->unpack_after_coarsening_and_refinement();
    }

  private:
    /// Handler managing the locally owned obstacles in the domain.
    std::unique_ptr<dealii::Particles::ParticleHandler<dim>> obstacle_handler;

    /// Property pool containing the properties of all global obstacles, stored locally on each
    /// MPI rank.
    mutable std::unique_ptr<dealii::Particles::PropertyPool<dim>> properties_global_obstacles;

    std::map<typename dealii::Triangulation<dim>::cell_iterator,
             std::vector<dealii::Particles::ParticleIterator<dim>>>
      cell_to_locally_owned_particle_cache;

    std::map<typename dealii::Triangulation<dim>::cell_iterator,
             std::vector<typename dealii::Particles::PropertyPool<dim>::Handle>>
      cell_to_ghost_particle_cache;

    /// MPI communicator used for synchronizing obstacle data across all ranks.
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;

    /// Timer data for profiling the obstacle search operations.
    dealii::TimerOutput &timer;

    /// The level of the triangulation at which particles are stored.
    int level_to_store_particles = 0;

    /// Variable to help keeping track of the maximum radius of the particles in the domain, which
    /// is relevant for determining the level at which to store particles.
    number max_particle_radius = 0;

    /**
     * @brief Deregisters all particles from the global obstacle property pool.
     */
    void
    deregister_property_pool() const;

    void
    sort_particles_into_local_level_cells();

    void
    communicate_ghost_particles();

    // similar to dealii::Particles::Particle functionality but also sends cell information on level
    // of interest as well as returns the handler in the new property pool when storing the
    // received particles.
    void *
    write_particle_data_to_memory(
      void                                                     *data_pointer,
      const dealii::Particles::ParticleIterator<dim>            particle,
      const typename dealii::Triangulation<dim>::cell_iterator &cell) const;

    struct ReceivedParticleData
    {
      typename dealii::Particles::PropertyPool<dim>::Handle handle;
      int                                                   cell_level;
      int                                                   cell_index;
    };


    ReceivedParticleData
    read_particle_data_from_memory(void                                 *data_pointer,
                                   dealii::Particles::PropertyPool<dim> &property_pool,
                                   const unsigned                        n_properties) const;

    std::size_t
    serialized_size_in_bytes(unsigned int n_properties) const;

    dealii::Triangulation<dim> const *triangulation;

    LevelCellPartitioner<dim> level_cell_partitioner;

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
  };

  template <int dim, typename number, typename ObstacleType>
  ObstacleDataStructure<dim, number>
  obstacle_data_structure_factory(const ObstacleDataStructureType   data_structure_type,
                                  const dealii::Triangulation<dim> &triangulation,
                                  const dealii::Mapping<dim>       &mapping,
                                  dealii::TimerOutput              &timer)
  {
    switch (data_structure_type)
      {
        case ObstacleDataStructureType::CompleteDomainSearch:
          return ObstacleDataStructure<dim, number>(
            ObstacleCompleteDomainSearch<dim, number, ObstacleType>(triangulation, mapping, timer));
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
  }
} // namespace MeltPoolDG
