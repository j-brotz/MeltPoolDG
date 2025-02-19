#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/numerics/rtree.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/level_set/nearest_point_data.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <vector>


namespace MeltPoolDG::LevelSet::Tools
{
  /**
   * @brief Compute nearest points to the isocontour of a level set function.
   *
   * This class provides methods to compute nearest points from a cloud of query points
   * (typically nodal locations from a DoFHandler) to an implicitly defined interface
   * represented by a level set (signed distance) function.
   *
   * The interface is approximated using a discrete representation, and the nearest points
   * are determined using one of several algorithms specified via @p NearestPointType.
   *
   * Supported methods:
   * - @p nearest_point: Discretizes the surface using the marching cubes algorithm and returns
   *   the closest point from the input cloud.
   * - @p nearest_point_fast: Same as @p nearest_point, but uses ArborX for efficient distributed
   *   nearest neighbor search.
   * - @p closest_point_normal: Iteratively refines the nearest point using gradient descent along
   *   the normal direction of the level set.
   * - @p closest_point_normal_collinear: Variant of the above that ensures the correction is
   *   collinear to the interface.
   * - @p closest_point_normal_collinear_coquerelle: Variant using the algorithm from
   *   Coquerelle and Glockner (2014), designed to preserve collinearity.
   */
  template <int dim, typename number>
  class NearestPoint
  {
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    /**
     * @brief Constructor.
     *
     * Initializes the nearest point projection utility.
     *
     * @param mapping Mapping used for geometry transformation.
     * @param dof_handler_signed_distance DoFHandler associated with the signed distance (level set) field.
     * @param signed_distance Vector containing the signed distance values.
     * @param normal_vector Normal vectors at the interface. Not needed for @p nearest_point and @p nearest_point_fast.
     * @param remote_point_evaluation Object used for evaluating points across MPI ranks.
     * @param nearest_point_data Struct containing configuration parameters for the projection.
     * @param timer_output Optional timer for profiling.
     */
    NearestPoint(const dealii::Mapping<dim>    &mapping,
                 const dealii::DoFHandler<dim> &dof_handler_signed_distance,
                 const VectorType              &signed_distance,
                 const BlockVectorType         &normal_vector,
                 dealii::Utilities::MPI::RemotePointEvaluation<dim, dim>   &remote_point_evaluation,
                 const NearestPointData<number>                            &nearest_point_data,
                 std::optional<std::reference_wrapper<dealii::TimerOutput>> timer_output = {});

    /**
     * @brief Initialize projection by computing nearest points for each DoF in the given DoFHandler.
     *
     * This function performs the projection of nodal locations from @p dof_handler_src_in to the
     * interface defined by the level set function. Depending on the algorithm type, this may
     * involve nearest neighbor queries or iterative corrections.
     *
     * If the source DoFHandler corresponds to a CutFEM method, you must also provide a destination
     * DoFHandler @p dof_handler_dst_in which must have the same unit support points and number of
     * components.
     *
     * @param dof_handler_src_in The source DoFHandler (used to define the nodal stencil points).
     * @param dof_handler_dst_in The optional destination DoFHandler (used for CutFEM or non-matching cases).
     */
    void
    reinit(const dealii::DoFHandler<dim> *dof_handler_src_in,
           const dealii::DoFHandler<dim> *dof_handler_dst_in = nullptr);

    /**
     * @brief Get the computed nearest points corresponding to the nodal locations.
     *
     * This function returns the result of the last call to reinit(). Each point corresponds
     * to a nodal location from the source DoFHandler within a narrow band around
     * the requested level set isocontour.
     *
     * @note Make sure that you have called reinit() before.
     *
     * @return A vector of projected points on the interface.
     */
    const std::vector<dealii::Point<dim>> &
    get_points() const;

    /**
     * @brief Get the global DoF indices associated with each nearest point.
     *
     * For each point in the stencil, this function provides the set of DoF indices (per component)
     * in the source FE space that the projected values correspond to.
     *
     * @note You must call reinit() before using this method.
     *
     * @return A vector of vectors of global DoF indices.
     */
    const std::vector<std::vector<dealii::types::global_dof_index>> &
    get_dof_indices() const;

    /**
     * @brief Populate a DoF vector with values from the interface.
     *
     * This function takes a DoF vector @p solution_in and interpolates its values at the previously
     * projected points on the interface. These values are then inserted into @p solution_out
     * at the corresponding DoFs.
     *
     * You can optionally apply a unary operation (e.g. scaling, filtering) to each value before
     * insertion. If @p zero_out is true, @p solution_out is reset to zero before writing.
     *
     * @tparam n_components Number of components (default is 1).
     * @param solution_out Output vector to write interface values into.
     * @param solution_in Input vector from which values are sampled.
     * @param zero_out Whether to zero the output vector before writing.
     * @param operation Optional function applied to each sampled value.
     */
    template <int n_components = 1>
    void
    fill_dof_vector_with_point_values(
      VectorType                                &solution_out,
      const VectorType                          &solution_in,
      const bool                                 zero_out  = false,
      const std::function<number(const number)> &operation = {}) const;

    /**
     * @brief Write the computed nearest points to a file.
     *
     * Dumps the list of nearest points (computed via reinit()) to a plain text table file
     * with the given filename. Each line contains the coordinates of a point.
     *
     * @param filename Name of the output file (default: "unmatched_points").
     */
    void
    write_to_file(const std::string filename = "unmatched_points") const;

    /// List of points where the projection failed or did not converge (for debug purposes).
    mutable std::vector<dealii::Point<dim>> points_not_found;

  private:
    /// Mapping
    const dealii::Mapping<dim> &mapping;

    /// DoFHandler associated with the level set function (signed distance field)
    const dealii::DoFHandler<dim> &dof_handler_ls;

    /// Level set / signed distance function used to define the interface
    const VectorType &signed_distance;

    /// Optional vector containing the interface normals at DoFs (used for projection direction or
    /// correction)
    const BlockVectorType &normal_vector;

    /// Parameters controlling the nearest point projection (e.g., tolerance, contour value,
    /// subdivisions, etc.)
    const NearestPointData<number> &nearest_point_data;

    /// Object for evaluating remote points across MPI ranks (used to gather values at projected
    /// positions)
    dealii::Utilities::MPI::RemotePointEvaluation<dim, dim> &remote_point_evaluation;

    // Tolerance to be reached for the distance of the projected points to the distance = 0
    // isosurface
    const number tol_distance;

    /// Threshold defining the narrow band for projection — usually a fraction of max(|distance|).
    /// Prevents projection far from the interface in regions where distance field may saturate.
    /// In the default case, we limit the interval for closest point projection to
    /// max(distance)*0.9999 to avoid projection in regions, where the distance is constant at
    /// max(distance).
    const number narrow_band_threshold;

    /// Tolerance for accuracy of the normal vector, if used in projection
    const number tolerance_normal_vector;

    /// Communicator associated with this projection process
    const MPI_Comm mpi_comm;

    dealii::ConditionalOStream pcout;
    /// Optional scoped timer for profiling different stages of projection
    std::optional<std::reference_wrapper<dealii::TimerOutput>> timer_output;

    /// Source DoFHandler: values are projected from this FE space
    const dealii::DoFHandler<dim> *dof_handler_src = nullptr;

    /// Optional destination DoFHandler: used when projecting between different FE spaces (e.g.
    /// cutFEM). Required when `src_is_cut = true`.
    const dealii::DoFHandler<dim> *dof_handler_dst = nullptr;

    // --- Output data: projection results ---

    /// Projected physical coordinates on the interface, corresponding to each DoF in the stencil
    std::vector<dealii::Point<dim>> projected_points_at_interface;

    /// Global DoF indices (per stencil point) — maps each stencil entry to its target DoFs
    std::vector<std::vector<dealii::types::global_dof_index>> dof_indices;

    /// Coordinates (physical space) of the stencil points used as nearest point queries
    std::vector<dealii::Point<dim>> stencil;

    // --- Internal state ---

    /// Temporary variable used to count total projected points in RPE or partitioning
    int total_points_rpe = 0;

    /// Tracks whether the `reinit()` function has been called, to prevent misuse
    bool is_reinit_called = false;

    /// Indicates whether the input vector corresponds to a cut DoFHandler (e.g. in embedded
    /// methods)
    bool input_vector_is_cut = false;

    // --- Internal data for fast path (ArborX-based projection) ---

    /// Noncontiguous partitioner used to communicate surface point values across MPI ranks
    dealii::Utilities::MPI::NoncontiguousPartitioner partitioner;

    /// Global indices of surface/interface points owned by this MPI rank
    std::vector<dealii::types::global_dof_index> locally_owned_surface_indices;

    /// Global indices of surface/interface points owned remotely (ghost points) corresponding to
    /// the stencil point owned by this rank
    std::vector<dealii::types::global_dof_index> ghost_surface_indices;

    /// Stores pairs of (cell level, cell index) with corresponding unit reference points on the
    /// interface. Used to evaluate FE values at interface locations.
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<dealii::Point<dim>>>>
      surface_cells_and_unit_points;

    /**
     * @brief Populate a DoF vector with values from the interface using ArborX acceleration.
     *
     * This internal function populates a DoF vector `solution_out` by sampling values from
     * `solution_in` at the interface using the fast nearest-point projection approach based on
     * ArborX. Compared to `fill_dof_vector_with_point_values()`, this method leverages spatial
     * acceleration structures for improved performance.
     *
     * @tparam n_components Number of vector components to extract (default is 1).
     * @param solution_out Output vector to be filled with interpolated values at the interface.
     * @param solution_in Input vector from which to sample the values.
     * @param zero_out If true, the `solution_out` vector is zeroed before values are written.
     * @param operation Optional unary operation to apply to each sampled value (e.g., scaling).
     */
    template <int n_components = 1>
    void
    fill_dof_vector_nearest_point_fast(
      VectorType                                &solution_out,
      const VectorType                          &solution_in,
      const bool                                 zero_out  = false,
      const std::function<number(const number)> &operation = {}) const;

    /**
     * @brief Clear all internally cached data.
     *
     * This is typically called during reinitialization to ensure that outdated
     * or invalid cached data structures (e.g., support points, DoF indices)
     * are properly reset before reuse.
     */
    void
    clear_cached_data();

    /**
     * @brief Register DoFHandler objects for source and destination spaces.
     *
     * Registers the degrees of freedom handlers used for sampling (`dof_handler_src_in`)
     * and for interpolation or projection (`dof_handler_dst_in`). These define the finite
     * element spaces used during interface value transfer.
     *
     * @param dof_handler_src_in Pointer to the DoFHandler used to extract interface values.
     * @param dof_handler_dst_in Pointer to the DoFHandler used to store the resulting values.
     */
    void
    register_dof_handlers(const dealii::DoFHandler<dim> *dof_handler_src_in,
                          const dealii::DoFHandler<dim> *dof_handler_dst_in);

    /**
     * @brief Collect support points from the source DoFHandler near the level set interface.
     *
     * Identifies and stores support points within a narrow band around the level set interface.
     * The width of this band is determined by an internal parameter `narrow_band_threshold`.
     * These points are used in the projection step to determine closest points.
     */
    void
    collect_narrow_band_support_points();

    /**
     * @brief Run the projection algorithm to determine nearest points on the interface.
     *
     * Executes the projection procedure using precomputed support points and the
     * ArborX spatial search structures. The resulting coordinates of the closest
     * points to the discrete interface are stored in `projected_points_at_interface`.
     */
    void
    run_projection_algorithm();

    /**
     * Perform a closest point projection to the surface by an iterative correction procedure
     * in the normal direction according to:
     *
     *   (k+1)   (k)     /  (k) \    /  (k) \
     * y      = y    - d | y    | nΓ | y    |    for k=0...max_iter
     *                   \      /    \      /
     *
     * with y being the closest point of a support point, d the signed distance function and
     * nΓ the interface normal vector. The iteration is skipped once the required tolerance for
     * d is achieved for all points within the input/output list of points @p y.
     */
    bool
    local_compute_normal_correction(std::vector<dealii::Point<dim>> &y);

    /**
     * Perform a closest point projection of a point x to the surface by an iterative correction
     * procedure in the normal direction and the tangential direction according to
     *
     * M. Coquerelle, S. Glockner (2014). A fourth-order accurate curvature computation in a
     * level set framework for two-phase flows subjected to surface tension forces. First,
     * the point is corrected in tangential direction via
     *
     *  (0)    (0)     /  (0) \       /  (0) \
     * y    = y    - ω | y    |  tΓ   | y    |
     *                i\      /    i  \      /
     *
     * with the tangential vector to the interface tΓ and the tangential distance ω
     *
     *      /  (0)    \                            2D: {0}
     * ω  = | y   - x | · tΓ             for i in
     *  i   \         /     i                      3D: {0,1}
     *
     * and subsequently is iteratively corrected in normal direction
     *
     *   (k+1)   (k)         /  (k) \
     * y      = y    - d  nΓ | y    |    for k=0...max_iter
     *                       \      /
     *
     * with y being the closest point of a support point, d the signed distance function and
     * nΓ the interface normal vector. The iteration is finished once the required tolerance for
     * d is achieved for all points within the input/output list of points @p y.
     */
    bool
    local_compute_normal_and_tangential_correction_coquerelle(std::vector<dealii::Point<dim>> &y);

    /**
     * Perform a closest point projection of a point x to the surface by an iterative correction
     * procedure in the normal direction and the tangential direction. First, we iteratively correct
     * in normal direction
     *
     *   (k+1)   (k)         /  (k) \
     * y      = y    - d  nΓ | y    |    for k=0...max_iter
     *                       \      /
     *
     * with y being the closest point of a support point, d the signed distance function and
     * nΓ the interface normal vector. The iteration is finished once the required tolerance for
     * d is achieved. Then, the algorithm continues with the tangential correction step
     *
     *  (k+1)    (k+1)  /  (k+1) \       /  (k+1) \
     * y     = y    - ω | y      |  tΓ   | y      |
     *                 i\        /    i  \        /
     *
     * with the tangential vector to the interface tΓ and the tangential distance ω
     *
     *      /  (k)    \                            2D: {0}
     * ω  = | y   - x | · tΓ             for i in
     *  i   \         /     i                      3D: {0,1}
     *
     */
    bool
    local_compute_normal_and_tangential_correction(std::vector<dealii::Point<dim>> &y);

    /**
     * Create a surface mesh and identify the closest point as the nearest vertex of the surface
     * mesh.
     */
    void
    local_compute_nearest_point();

    /**
     * Fast path -- create a surface mesh and identify the closest point as the nearest vertex of
     * the surface mesh.
     */
    void
    local_compute_nearest_point_fast();
  };
} // namespace MeltPoolDG::LevelSet::Tools
