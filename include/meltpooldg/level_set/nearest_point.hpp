#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/level_set/nearest_point_data.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>


namespace MeltPoolDG::LevelSet::Tools
{
  /**
   * Compute nearest points to the isocontour of a level set function
   *
   * Based on a level set function and a cloud of points in the domain (stencil), compute
   * the corresponding nearest points to a discrete representation of the interface (isocontour).
   * The stencil is computed considering nodal points of a DoFHandler lying within a narrow band.
   * We support four types of algorithms (@p NearestPointType):
   *    - nearest_point: discretize the surface via the marching cube algorithm and take the closest
   *      point to the surface (cheap!)
   *    - closest_point_normal: iteratively correct the nearest point following the normal direction
   *      of the point.
   *    - closest_point_normal_collinear: extension of closest_point_normal to also ensure that the
   *      closest point is collinear to the interface (standard algorithm)
   *    - closest_point_normal_collinear_coquerelle: extension of closest_point_normal to also
   *      ensure that the closest point is collinear to the interface (algorithm proposed by
   *      Coquerelle and Glockner (2014))
   */
  template <int dim, typename number>
  class NearestPoint
  {
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

  public:
    /**
     * Constructor
     *
     * @param mapping Mapping of the geometry.
     * @param dof_handler_signed_distance DoFHandler of the level set/signed distance function.
     * @param signed_distance Vector of the level set/signed distance function.
     * @param remote_point_evaluation Cache for MPI::RemotePointEvaluation.
     * @param additional_data Parameters for calculating nearest point.
     */
    NearestPoint(const dealii::Mapping<dim>    &mapping,
                 const dealii::DoFHandler<dim> &dof_handler_signed_distance,
                 const VectorType              &signed_distance,
                 const BlockVectorType         &normal_vector,
                 dealii::Utilities::MPI::RemotePointEvaluation<dim, dim>   &remote_point_evaluation,
                 const NearestPointData<number>                            &additional_data,
                 std::optional<std::reference_wrapper<dealii::TimerOutput>> timer_output = {});

    /**
     * Update the nearest points of the nodal points from a given DoFHandler @p dof_handler_src_in.
     * Note: @p dof_handler_req may be a CutFEM DofHandler. If so, you must provide a @p
     * dof_handler_dst_in for the destination vector, which must have the identical unit support
     * points and number of components as dof_handler_src_in.
     */
    void
    reinit(const dealii::DoFHandler<dim> *dof_handler_src_in,
           const dealii::DoFHandler<dim> *dof_handler_dst_in = nullptr);

    /**
     * Getter function for the nearest points, corresponding to the nodal points of the
     * DoFHandler passed into the reinit() function.
     *
     * @note Make sure that you have called reinit() before.
     */
    const std::vector<dealii::Point<dim>> &
    get_points() const;

    /**
     * Getter function for the DoF indices of the nearest points, corresponding to the nodal points
     * of the source DoFHandler passed into the reinit() function.
     *
     * @note Make sure that you have called reinit() before.
     */
    const std::vector<std::vector<dealii::types::global_dof_index>> &
    get_dof_indices() const;

    /**
     * For a given DoF vector @p solution_in (according to the layout of the DoFHandler passed into
     * the reinit function), take the value at the requested isocontour (defined by closest point
     * projection data), distribute it over the interfacial region and store it into @p solution_out.
     * Via @p operation, the value at the interface from @p solution_in can be manipulated prior
     * to store it into @p solution_out.
     * Set @p zero_out to true @p if solution_out should be set to zero in advance.
     *
     * @note Make sure that you have called reinit() before.
     */
    template <int n_components = 1>
    void
    fill_dof_vector_with_point_values(
      VectorType                                &solution_out,
      const VectorType                          &solution_in,
      const bool                                 zero_out  = false,
      const std::function<number(const number)> &operation = {}) const;


    /**
     * Write the nearest points, calculated via reinit(), to a table file @p filename.
     */
    void
    write_to_file(const std::string filename = "unmatched_points") const;

    mutable std::vector<dealii::Point<dim>> points_not_found;

  private:
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

    const dealii::Mapping<dim>     &mapping;
    const dealii::DoFHandler<dim>  &dof_handler_ls;
    const VectorType               &signed_distance;
    const BlockVectorType          &normal_vector;
    const NearestPointData<number> &additional_data;

    dealii::Utilities::MPI::RemotePointEvaluation<dim, dim> &remote_point_evaluation;

    // Tolerance to be reached for the distance of the projected points to the distance = 0
    // isosurface
    const number tol_distance;
    // In the default case, we limit the interval for closest point projection to
    // max(distance)*0.9999 to avoid projection in regions, where the distance is constant at
    // max(distance).
    const number narrow_band_threshold;

    const number tolerance_normal_vector;

    const MPI_Comm mpi_comm;

    dealii::ConditionalOStream pcout;

    std::optional<std::reference_wrapper<dealii::TimerOutput>> timer_output;

    // DoFHandler of the source vector
    const dealii::DoFHandler<dim> *dof_handler_src = nullptr;
    // optional DoFHandler of the destination vector. If src_is_cut = true, this is not optional
    const dealii::DoFHandler<dim> *dof_handler_dst = nullptr;

    // vectors to be filled: projected points to the interface corresponding to DoF indices
    std::vector<dealii::Point<dim>>                           projected_points_at_interface;
    std::vector<std::vector<dealii::types::global_dof_index>> dof_indices;
    std::vector<dealii::Point<dim>>                           stencil;

    bool is_reinit_called    = false;
    bool input_vector_is_cut = false;

    // this is just a temporary variable to be called within the projection operators
    int total_points_rpe = 0;
  };
} // namespace MeltPoolDG::LevelSet::Tools
