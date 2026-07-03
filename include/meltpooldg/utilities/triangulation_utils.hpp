#pragma once

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace MeltPoolDG
{
  enum TriangulationType
  {
    shared,
    parallel_distributed,
    parallel_fullydistributed,
    serial
  };

  template <int dim, int spacedim = dim>
  TriangulationType
  get_triangulation_type(const dealii::Triangulation<dim, spacedim> &tria);

  /**
   * A class which caches the adjacent cells for each cell on a given level of a triangulation.
   * Adjacency is defined as vertex-sharing, i.e., two cells are considered adjacent if they share
   * at least one vertex. The cache is built for a specific level of the triangulation and can be
   * queried for the adjacent cells of any cell on that level, assuming the cell is locally
   * available (i.e., it is either locally owned, ghost, or artificial).
   *
   * @note The class caches iterators to cells, which are only valid as long as the triangulation is
   * not modified. If the triangulation is modified (e.g., by refining or coarsening), the cache
   * must be rebuilt by calling the build_cache() function again.
   */
  template <int dim>
  class LevelAdjacentCellsCache
  {
    using tria_iterator = dealii::TriaIterator<dealii::CellAccessor<dim>>;

  public:
    /**
     * Build the adjacency cache for the given level of the triangulation.
     *
     * @param tria The triangulation for which to build the cache.
     * @param level The level for which to build the cache.
     */
    void
    build_cache(const dealii::Triangulation<dim> &tria, const unsigned int level);

    /**
     * Get the adjacent cells of a given cell on the specified level.
     *
     * @param cell The cell for which to get the adjacent cells.
     *
     * @note The returned set of adjacent cells does not include the cell itself.
     * @note If an adjacent cell is not available on the same level, the parent cell of the
     * hypothesized adjacent cell on the specified level is returned instead.
     */
    const std::set<dealii::TriaIterator<dealii::CellAccessor<dim>>> &
    get_adjacent_cells(const tria_iterator &cell) const;

    /**
     * Get the total number of cells on the specified level across all MPI ranks.
     */
    unsigned int
    n_global_cells_on_level() const;

  private:
    /// The total number of cells on the specified level across all MPI ranks.
    unsigned int n_global_level_cells = 0;

    /// A map that assigns to each cell on the specified level the set of its adjacent cells. Note,
    /// that this does not include the cell itself.
    std::map<tria_iterator, std::set<tria_iterator>> adjacent_cells_cache;

    /// A boolean indicating whether the adjacency cache has been built, i.e., whether the
    /// build_cache() function has been called.
    bool is_cache_built = false;

    /// The level for which the adjacency cache has been built.
    unsigned int cache_level = 0;

    /**
     * Assert that the adjacency cache has been built before accessing it. If the cache has not been
     * built, an exception is thrown. This is checked by the is_cache_built boolean variable.
     */
    void
    assert_cache_built() const;

    /**
     * Build the adjacency cache for the given level of the triangulation. This function populates
     * the adjacency cache with the adjacent cells for each cell on the specified level.
     *
     * @param tria The triangulation for which to build the cache.
     * @param level The level for which to build the cache.
     */
    void
    cache_adjacent_cells(const dealii::Triangulation<dim> &tria, const unsigned int level);

    /**
     * Compute the total number of cells on the specified level across all MPI ranks.
     *
     * @param tria The triangulation for which to compute the cell count.
     * @param level The level for which to compute the cell count.
     */
    void
    compute_global_cell_count(const dealii::Triangulation<dim> &tria, const unsigned int level);
  };


  /**
   * @brief Computes and stores the MPI communication pattern for a given level of a distributed
   * triangulation.
   *
   * The pattern implemented in this class is designed such that each rank that owns an active cell
   * being a descendant of a cell on the specified level gets the information about that cell and
   * its adjacent cells on the same level from the rank owning the corresponding cell on the
   * specified level. Hereby, adjacency is defined as vertex-sharing, i.e., two cells are considered
   * adjacent if they share at least one vertex.
   *
   * This class is strictly designed to analyze the triangulation and determine the  communication
   * pattern (i.e., which cell IDs need to be sent to which ranks and which cell IDs are expected to
   * be received from which ranks). It does not perform the actual data communication itself.
   * Instead, it exposes this metadata blueprint via getter functions so that the actual
   * communication can be implemented elsewhere in the user-code.
   *
   * To understand the functionality of the different member functions, it is beneficial to
   * familiarize yourself with the following definitions:
   *
   * - **Partition-level**: The refinement level on the triangulation for which the communication
   * pattern is built. This level is specified by the user when calling the build_pattern()
   * function.
   *
   * - **Descendant cells**: The active cells on the triangulation that aredescendants of a cell on
   * the partition-level.
   *
   * - **Ancestor cells**: The cells on the partition-level that contain at least one locally owned
   * active descendant cell.
   *
   * - **Adjacent cells**: Cells on the same level that share at least one vertex with a given cell.
   *
   * - **Relevant cells**: Depend on the context of the cell being queried: For an ancestor cell on
   * the partition-level relevant cells are the ancestor cell itself and all adjacent cells on the
   * same level. For an active cell on the triangulation, the relevant cells are the ancestor cell
   * of that active cell and all adjacent cells on the same level.
   *
   * @note This class assumes that the relevant partition-level cells are available at least as
   * artificial cells on the current MPI rank, which is guaranteed if the triangulation was created
   * with the multigrid hierarchy enabled.
   */
  template <int dim>
  class LevelCommunicationPattern
  {
  public:
    LevelCommunicationPattern(const dealii::Triangulation<dim> &tria);

    /**
     * Build the communication pattern for the given level. This includes determining which cells
     * on the partition level are relevant for the local process and which other processes need to
     * receive data from or send data to the local process based on the adjacency of cells on the
     * partition level. The communication pattern is stored internally in the class and can be
     * accessed via the provided getter functions.
     *
     * @param level The level for which the communication pattern should be built.
     */
    void
    build_pattern(const unsigned int level);

    /**
     * Get a map that assigns to each MPI rank the cell ids of the locally available cells for
     * which the current process needs to send data to the corresponding other rank.
     */
    std::map<unsigned int, std::vector<dealii::CellId>>
    cells_to_send() const;

    /**
     * Get a map that assigns to each MPI rank the cell ids of locally available cells for which
     * the current process needs to receive data from the corresponding other rank.
     */
    std::map<unsigned int, std::vector<dealii::CellId>>
    cells_to_receive() const;

    /**
     * Return the number of processes to which the current process needs to send data.
     */
    unsigned
    n_processes_to_send_to() const
    {
      return rank_to_cells_send.size();
    }

    /**
     * Return the number of processes from which the current process needs to receive data.
     */
    unsigned
    n_processes_to_receive_from() const
    {
      return rank_to_cells_receive.size();
    }

    /**
     * Number of ranks to which data is sent.
     */
    unsigned int
    n_send_ranks() const;

    /**
     * Number of ranks from which data is received.
     */
    unsigned int
    n_receive_ranks() const;

    /**
     * MPI ranks to which data is sent.
     */
    std::vector<unsigned int>
    send_ranks() const;

    /**
     * MPI ranks from which data is received.
     */
    std::vector<unsigned int>
    receive_ranks() const;

  private:
    /**
     * Return the cells on the partition level that contain at least one locally owned active
     * descendant cell. The returned cells may be locally owned, ghost, or artificial cells.
     *
     * @return Cell ids of the relevant cells on the partition level.
     *
     * @note This function assumes that the relevant partition-level cells are available at least as
     * artificial cells on the current MPI rank. This is guaranteed if the triangulation was
     * created with the multigrid hierarchy enabled.
     */
    std::vector<dealii::CellId>
    owned_active_cells_ancestors() const;

    /**
     * Gather the relevant partition-level cells from all MPI ranks.
     *
     * @param local_ancestor_cells Cell ids of the partition-level cells relevant
     * to the current MPI rank.
     *
     * @return A vector containing, for each MPI rank, the corresponding relevant
     * partition-level cell ids.
     */
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
    gather_ancestors(const std::vector<dealii::CellId> &local_ancestor_cells) const;

    /**
     * Given a partition-level cell, determine all adjacent cells which are a rank local ancestor
     * cell.
     *
     * @param cell_id Cell id of the partition-level cell whose adjacent relevant cells should be
     * determined.
     * @param owned_active_cell_ancestors Cell ids of the partition-level cells containing locally
     * owned active descendant cells.
     * @param adjacent_cells_cache The cache of adjacent cells for the partition-level cells to
     * allow for efficient lookup of adjacent cells.
     *
     * @return Cell ids of the adjacent relevant cells.
     */
    std::vector<dealii::CellId>
    adjacent_relevant_cells(const dealii::CellId               &cell_id,
                            const std::vector<dealii::CellId>  &owned_active_cell_ancestors,
                            const LevelAdjacentCellsCache<dim> &adjacent_cells_cache) const;

    /**
     * For each MPI rank, determine the relevant rank local cells adjacent to the cells requested by
     * that rank. In this case adjacent cells include the cell itself and all cells on the same
     * level sharing at least one vertex with it.
     *
     * @param requested_cells_by_rank Vector containing, for each MPI rank, the corresponding
     * relevant partition-level cells they need to know about.
     * @param owned_active_cell_ancestors Cell ids of the partition-level cells containing locally
     * owned active descendant cells.
     *
     * @return A vector whose entries contain an MPI rank together with the corresponding adjacent
     * relevant local cells.
     */
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
    relevant_cells_for_ranks(
      const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &requested_cells_by_rank,
      const std::vector<dealii::CellId> &owned_active_cell_ancestors) const;

    /**
     * Exchange relevant cells with other MPI processes. Each process sends the relevant cells
     * adjacent to the requested partition-level cells to the corresponding MPI rank and receives
     * the relevant cells adjacent to the requested partition-level cells from the other MPI ranks
     *
     * @param cells_to_send A vector of pairs with the first element being the MPI rank and the
     * second element being the corresponding relevant cells for which the corresponding MPI
     * process will receive data from the current process.
     *
     * @return A vector of pairs with the first element being the MPI rank and the second element
     * being the corresponding relevant cells for which the current MPI process will receive data
     * from the corresponding MPI process.
     */
    std::vector<std::pair<unsigned, std::vector<dealii::CellId>>>
    exchange_relevant_cells(
      const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &cells_to_send) const;

    /**
     * Store the computed send and receive communication pattern.
     */
    void
    store_rank_cell_maps(
      const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &data_to_receive,
      const std::vector<std::pair<unsigned, std::vector<dealii::CellId>>> &data_to_send);

    /**
     * Build the MPI communication pattern for the current partition level.
     */
    void
    build_communication_pattern();

    /**
     * Assert that the communication pattern has been built before accessing it. If the pattern has
     * not been built, an exception is thrown.
     */
    void
    assert_pattern_built() const;

    /// Map that assigns to each MPI rank the cell ids of the locally available partition-level
    /// cells for which the current process needs to send data to the corresponding other rank.
    std::map<unsigned int, std::vector<dealii::CellId>> rank_to_cells_send;

    /// Map that assigns to each MPI rank the cell ids of locally available partition-level cells
    /// for which the current process needs to receive data from the corresponding other rank.
    std::map<unsigned int, std::vector<dealii::CellId>> rank_to_cells_receive;

    /// Reference to the triangulation for which the communication pattern is built.
    const dealii::Triangulation<dim> &triangulation;

    /// The cell level on which the communication pattern is built.
    unsigned int partition_level = 0;

    /// MPI communicator for the communication within the communication pattern. This is typically
    /// the same as the one of the triangulation, but it is stored here separately for better
    /// readability.
    const MPI_Comm mpi_communicator;

    /// A boolean indicating whether the communication pattern has been built, i.e., whether
    // the build_pattern() function has been called.
    bool is_pattern_built = false;
  };
} // namespace MeltPoolDG
