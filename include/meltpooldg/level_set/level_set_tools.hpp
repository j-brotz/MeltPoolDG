#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>

#include <functional>
#include <tuple>
#include <utility>
#include <vector>


namespace MeltPoolDG::LevelSet::Tools
{
  enum BooleanType
  {
    Union,
    Intersection,
    Subtraction
  };

  /**
   * Interpolate between @p val1 and @p val2 with the following function
   *
   * x = (1 - ls) val1 + ls val2
   */
  template <typename value_type1, typename value_type2, typename value_type3>
  inline value_type1
  interpolate(const value_type1 &ls, const value_type2 &val1, const value_type3 &val2)
  {
    return (1. - ls) * val1 + ls * val2;
  }

  /**
   * Interpolate between @p val1 and @p val2 with the reciprocal function
   *
   *             1
   * x = ---------------------
   *       (1 - ls)      ls
   *      ---------- + ------
   *         val1       val2
   */
  template <typename value_type1, typename value_type2, typename value_type3>
  inline value_type1
  interpolate_reciprocal(const value_type1 &ls, const value_type2 &val1, const value_type3 &val2)
  {
    // clang-format off
      return                    1.
             / // --------------------------------
                   ((1. - ls) / val1 + ls / val2);
    // clang-format on
  }

  /**
   * Interpolate between @p val1 and @p val2 with the cubic function
   *
   * x = val1 + ( val2 - val1 ) ( -2 ls³ + 3 ls² )
   */
  template <typename value_type1, typename value_type2, typename value_type3>
  inline value_type1
  interpolate_cubic(const value_type1 &ls, const value_type2 &val1, const value_type3 &val2)
  {
    return val1 + (val2 - val1) * (-2. * ls * ls * ls + 3. * ls * ls);
  }

  /**
   * Derivative of interpolate_cubic() with respect to @ls. Returns
   *
   * ( val2 - val1 ) (-6 ls² + 6 ls)
   */
  template <typename value_type1, typename value_type2, typename value_type3>
  inline value_type1
  interpolate_cubic_derivative(const value_type1 &ls,
                               const value_type2 &val1,
                               const value_type3 &val2)
  {
    return (val2 - val1) * (-6. * ls * ls + 6. * ls);
  }


  /**
   * For two indicator vectors, representing e.g. implicit geometries, this function computes a
   * boolean operation and returns the resulting vector. The user has to take care on distributing
   * relevant constraints afterwards.
   */
  template <typename number>
  dealii::LinearAlgebra::distributed::Vector<number>
  merge_two_indicator_fields(const dealii::LinearAlgebra::distributed::Vector<number> &indicator_1,
                             const dealii::LinearAlgebra::distributed::Vector<number> &indicator_2,
                             BooleanType  type                     = BooleanType::Union,
                             const number indicator_value_interior = 1.0,
                             const number indicator_value_exterior = -1.0);

  /**
   * @brief Identifies volume cells intersected by a level set and extracts the corresponding
   *        quadrature points at the intersection (in both real and reference coordinates).
   *
   * This function applies a marching cube algorithm to detect where a level set function
   * intersects the computational mesh. For each locally-owned cell that is cut by the level set
   * interface (i.e., where the level set function crosses a given contour value), the function
   * computes the intersection points and maps them to unit (reference) cell coordinates.
   *
   * @param[out] surface_cells_and_unit_points
   *             A list of locally-owned cells that are intersected by the level set surface, each
   *             paired with its corresponding unit (reference) quadrature points.
   *
   * @param[out] surface_points
   *             A list of all real-space intersection points on the level set surface, collected
   * from all intersected volume cells.
   *
   * @param[in]  dof_handler
   *             The DoFHandler associated with the finite element field representing the level set.
   *
   * @param[in]  mapping
   *             The Mapping used to transform between real and unit cell coordinates.
   *
   * @param[in]  level_set_vector
   *             The distributed vector storing the level set function values over the mesh.
   *
   * @param[in]  contour_value
   *             The isocontour value that defines the level set interface (default: 0.0).
   *
   * @param[in]  n_subdivisions
   *             The number of subdivisions per cell edge for the marching cubes algorithm
   *             (used to refine intersection detection; default: 1).
   *
   * @param[in]  tolerance
   *             A small numerical tolerance used to determine zero-crossings in the marching cubes
   *             interface detection (default: 1e-10).
   */
  template <int dim, typename number>
  void
  collect_interface_cells_and_intersection_points(
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, std::vector<dealii::Point<dim>>>>
                                                             &surface_cells_and_unit_points,
    std::vector<dealii::Point<dim>>                          &surface_points,
    const dealii::DoFHandler<dim>                            &dof_handler,
    const dealii::Mapping<dim>                               &mapping,
    const dealii::LinearAlgebra::distributed::Vector<number> &level_set_vector,
    const number                                              contour_value  = 0.0,
    const unsigned int                                        n_subdivisions = 1,
    const number                                              tolerance      = 1e-10);

  /**
   * @brief Evaluates user-defined quantities at an implicitly defined interface.
   *
   * This utility function enables the evaluation of quantities at interfaces defined
   * implicitly by a scalar field (level set function), represented by the
   * @p level_set_vector. The interface is identified as the contour where the
   * level set function equals the specified @p contour_value.
   *
   * The interface is reconstructed on each active cell using the Marching Cubes
   * algorithm (or an optional alternative method if @p use_mca is false). The
   * algorithm identifies quadrature points within the reference cell and computes
   * corresponding surface integration weights (JxW values) for numerical integration.
   *
   * For each active cell in the provided DoFHandler cut by the interface, the
   * user-supplied lambda function @p evaluate_at_interface_points is called.
   *
   * This interface evaluation can be used to implement a variety of tasks,
   * such as integrating quantities over the interface, interpolating field
   * values, or assembling contributions into global data structures like DoF
   * vectors.
   *
   * @tparam dim The spatial dimension.
   * @tparam number The numeric type used for computations (e.g., float, double).
   *
   * @param[in] dof_handler The DoFHandler associated with the finite element
   *                        space defining the level set function.
   * @param[in] mapping A Mapping object describing the transformation from
   *                    reference to real space.
   * @param[in] level_set_vector A vector representing the level set function,
   *                             defined at the DoF locations.
   * @param[in] evaluate_at_interface_points A user-defined callback function
   *                                         that is called for each active cell
   *                                         containing interface segments.
   *                                         The callback signature must be:
   *                                         ```
   *                                         void(const active_cell_iterator &cell,
   *                                              const std::vector<Point<dim>> &points_real,
   *                                              const std::vector<Point<dim>> &points_reference,
   *                                              const std::vector<number> &JxW)
   *                                         ```
   * @param[in] contour_value The level set value at which the interface is
   *                          extracted (default is 0.0).
   * @param[in] n_subdivisions The number of subdivisions per cell used to
   *                            increase the resolution of the marching cubes grid
   *                            (default is 1, i.e., no subdivision).
   * @param[in] tolerance A small tolerance used for numerical comparisons when
   *                      identifying interface segments (default is 1e-10).
   * @param[in] use_mca A flag indicating whether to use the Marching Cubes Algorithm
   *                    (MCA) for interface reconstruction (default is true).
   *                    If false, the NonMatching infrastructure of deal.II is used.
   *
   * @note The accuracy of the interface location depends on the resolution of the
   *       level set function and the number of subdivisions. For better accuracy,
   *       consider refining the mesh or increasing @p n_subdivisions.
   */
  template <int dim, typename number>
  void
  evaluate_at_interface(const dealii::DoFHandler<dim>                            &dof_handler,
                        const dealii::Mapping<dim>                               &mapping,
                        const dealii::LinearAlgebra::distributed::Vector<number> &level_set_vector,
                        const std::function<void(
                          const typename dealii::DoFHandler<dim>::active_cell_iterator & /*cell*/,
                          const std::vector<dealii::Point<dim>> & /*quadrature_points_real*/,
                          const std::vector<dealii::Point<dim>> & /*quadrature_points_reference*/,
                          const std::vector<number> & /*JxW*/)> &evaluate_at_interface_points,
                        const number                             contour_value  = 0.0,
                        const unsigned int                       n_subdivisions = 1,
                        const number                             tolerance      = 1e-10,
                        const bool                               use_mca        = true);

  /**
   * @brief Generates surface mesh data corresponding to an implicit interface.
   *
   * This function constructs a surface mesh representation of an interface
   * implicitly defined by a scalar field (level set function), given in
   * the form of a `level_set_as_heaviside` vector. The interface corresponds
   * to the isosurface where the level set equals the specified @p contour_value.
   *
   * Internally, this function uses the Marching Cubes algorithm (or an optional
   * alternative method) to identify interface segments on each active cell.
   * The resulting surface mesh information is returned as a list of tuples,
   * where each tuple contains:
   * - the cell iterator of the active cell containing a portion of the interface,
   * - the unit (reference space) quadrature points on the interface within that cell,
   * - and the corresponding (surface) integration weights for those points.
   *
   * This mesh information can be used for interface visualization or integration
   * over implicitly defined surfaces.
   *
   * @tparam dim The spatial dimension of the problem.
   * @tparam number The numeric type used in the computation (e.g., float, double).
   *
   * @param[in] dof_handler The DoFHandler associated with the finite element space
   *                        describing the level set field.
   * @param[in] mapping A Mapping object used to transform between reference and real space.
   * @param[in] level_set_as_heaviside A vector representing the level set field.
   * @param[in] contour_value The value at which to extract the interface (default is 0.0).
   * @param[in] n_subdivisions The number of subdivisions per cell for resolving
   *                            the interface (default is 1).
   * @param[in] tolerance Numerical tolerance for comparing level set values (default is 1e-10).
   * @param[in] use_mca If true, the Marching Cubes Algorithm is used for interface
   *                    reconstruction (default is true).
   *
   * @return A vector of tuples, each containing:
   *   - a cell iterator to an active dim-cell containing dim-1 interface geometry,
   *   - a list of quadrature points (in unit coordinates) lying on the interface,
   *   - and the associated integration weights (JxW values).
   */
  template <int dim, typename number>
  std::vector<std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                         std::vector<dealii::Point<dim>> /*unit_points*/,
                         std::vector<number> /*weights*/
                         >>
  generate_surface_mesh_info(const dealii::DoFHandler<dim>                            &dof_handler,
                             const dealii::Mapping<dim>                               &mapping,
                             const dealii::LinearAlgebra::distributed::Vector<number> &level_set,
                             const number       contour_value  = 0.0,
                             const unsigned int n_subdivisions = 1,
                             const number       tolerance      = 1e-10,
                             const bool         use_mca        = true);

  /**
   * This utility function computes a point cloud @p global_points_normal_to_interface
   * in a narrow band around a level set vector. The parameter
   * @p global_points_normal_to_interface_pointer holds at indices [n, n+1] the index
   * range of connected points along the normal corresponding to the point n at the
   * interface.
   * First, the marching cube algorithm is exploited to determine points at the interface
   * given at the contour level @p contour_value. Then, for each point at the interface
   * points along the normal are generated.
   */
  template <int dim, typename number>
  void
  generate_points_along_normal(
    std::vector<dealii::Point<dim>> &global_points_normal_to_interface,
    std::vector<unsigned int>       &global_points_normal_to_interface_pointer,
    const dealii::DoFHandler<dim>   &dof_handler_ls,
    const dealii::FESystem<dim>     &fe_normal,
    const dealii::Mapping<dim>      &mapping,
    const dealii::LinearAlgebra::distributed::Vector<number>      &level_set_vector,
    const dealii::LinearAlgebra::distributed::BlockVector<number> &normal_vector,
    const number                                                   max_distance_per_side,
    const unsigned int                                             n_inc_per_side,
    const bool                                                     bidirectional      = true,
    const number                                                   contour_value      = 0.0,
    const unsigned int                                             n_subdivisions_MCA = 1);

  /**
   * Set the material ID of cells depending on their level-set values, given by
   * @p level_set_heaviside and the corresponding DoFHandler index @p ls_dof_idx.
   * Cells with level-set values larger than or equal to the threshold value
   * (@p lower_threshold) are indicated by a material_id of 1, others by
   * a material_id of 0.
   *
   * @note This function should only be used, if the isosurface is aligned with
   * the cell faces, because we do not treat real cut-cells special.
   */
  template <int dim, typename number>
  void
  set_material_id_from_level_set(
    const ScratchData<dim, dim, number>                      &scratch_data,
    const unsigned int                                        ls_dof_idx,
    const dealii::LinearAlgebra::distributed::Vector<number> &level_set_heaviside,
    const number                                              lower_threshold = 0.5);

  template <typename number>
  dealii::VectorizedArray<number>
  compute_mask_narrow_band(const dealii::VectorizedArray<number> &val,
                           const number                           narrow_band_threshold)
  {
    dealii::VectorizedArray<number> indicator = 1.0;
    for (unsigned int v = 0; v < dealii::VectorizedArray<number>::size(); ++v)
      if (std::abs(val[v]) >= narrow_band_threshold)
        indicator[v] = 0.0;

    return indicator;
  }
} // namespace MeltPoolDG::LevelSet::Tools
