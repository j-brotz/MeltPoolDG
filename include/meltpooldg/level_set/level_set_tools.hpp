#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>


namespace dealii::GridGenerator
{
  template <int dim, typename number>
  void
  create_triangulation_with_marching_cube_algorithm(
    Triangulation<dim - 1, dim>                              &tria,
    const dealii::Mapping<dim>                               &mapping,
    const DoFHandler<dim>                                    &background_dof_handler,
    const dealii::LinearAlgebra::distributed::Vector<number> &ls_vector,
    const number                                              iso_level,
    const unsigned int                                        n_subdivisions = 1,
    const number                                              tolerance      = 1e-10)
  {
    std::vector<dealii::Point<dim>> vertices;
    std::vector<CellData<dim - 1>>  cells;
    SubCellData                     subcelldata;

    const GridTools::MarchingCubeAlgorithm<dim, dealii::LinearAlgebra::distributed::Vector<number>>
      mc(mapping, background_dof_handler.get_fe(), n_subdivisions, tolerance);

    const bool vector_is_ghosted = ls_vector.has_ghost_elements();

    if (vector_is_ghosted == false)
      ls_vector.update_ghost_values();

    mc.process(background_dof_handler, ls_vector, iso_level, vertices, cells);

    if (vector_is_ghosted == false)
      ls_vector.zero_out_ghost_values();

    std::vector<unsigned int> considered_vertices;

    // note: the following operation does not work for simplex meshes yet
    // GridTools::delete_duplicated_vertices (vertices, cells, subcelldata,
    // considered_vertices);

    if (vertices.size() > 0)
      tria.create_triangulation(vertices, cells, subcelldata);
  }
} // namespace dealii::GridGenerator


namespace MeltPoolDG::LevelSet::Tools
{
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
                             const number indicator_value_exterior = -1.0)
  {
    AssertThrow(indicator_1.size() == indicator_2.size(),
                dealii::ExcMessage(
                  "The two level set vectors to be merged must be of equal length."));

    AssertThrow(indicator_value_interior != indicator_value_exterior,
                dealii::ExcMessage(
                  "The indicator value in the interior of the implicit geometry must be "
                  "different than the one in the exterior. Make sure that the function "
                  "arguments indicator_value_interior and indicator_value_exterior are "
                  "set correctly."));

    AssertThrow(indicator_1.linfty_norm() ==
                  std::max(indicator_value_exterior, indicator_value_interior),
                dealii::ExcMessage(
                  "The maximum indicator value does not correspond to the given bounds "
                  "of indicator_value_interior and indicator_value_exterior. Make sure that "
                  "the indicator_value_interior and indicator_value_exterior comply with the "
                  "indicator field."));

    dealii::LinearAlgebra::distributed::Vector<number> merge_indicator(indicator_1);

    bool is_interior_larger_than_exterior = indicator_value_interior > indicator_value_exterior;

    for (const auto &i : indicator_1.locally_owned_elements())
      switch (type)
        {
          case BooleanType::Union:
            merge_indicator[i] = is_interior_larger_than_exterior ?
                                   std::max(indicator_1[i], indicator_2[i]) :
                                   std::min(indicator_1[i], indicator_2[i]);
            break;
          case BooleanType::Intersection:
            merge_indicator[i] = is_interior_larger_than_exterior ?
                                   std::min(indicator_1[i], indicator_2[i]) :
                                   std::max(indicator_1[i], indicator_2[i]);
            break;
          case BooleanType::Subtraction:
            merge_indicator[i] =
              indicator_1[i] == indicator_2[i] && indicator_1[i] == indicator_value_interior ?
                indicator_value_exterior :
                indicator_1[i];
            break;
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }

    merge_indicator.compress(dealii::VectorOperation::insert);

    return merge_indicator;
  }

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
   * algorithm identifies interface points within the reference cell and computes
   * corresponding integration weights (JxW values) for numerical integration.
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
                          const std::vector<dealii::Point<dim>> & /*points_real*/,
                          const std::vector<dealii::Point<dim>> & /*points_reference*/,
                          const std::vector<number> & /*JxW*/)> &evaluate_at_interface_points,
                        const number                             contour_value  = 0.0,
                        const unsigned int                       n_subdivisions = 1,
                        const number                             tolerance      = 1e-10,
                        const bool                               use_mca        = true)
  {
    // data structures for marching-cube algorithm
    const dealii::QGauss<dim == 1 ? 1 : dim - 1> surface_quad(dof_handler.get_fe().degree + 1);

    dealii::GridTools::MarchingCubeAlgorithm<dim,
                                             dealii::LinearAlgebra::distributed::Vector<number>>
      mc(mapping,
         dof_handler.get_fe(), // todo
         n_subdivisions,
         tolerance);

    const bool is_ghosted = level_set_vector.has_ghost_elements();

    if (not is_ghosted)
      level_set_vector.update_ghost_values();

    // data structures for NonMatching::FEValues
    const unsigned int nm_n_q_points_1D =
      std::max<unsigned int>(dof_handler.get_fe().degree, n_subdivisions + 1);
    const dealii::hp::QCollection<dim> nm_quad(dealii::QGauss<dim>{nm_n_q_points_1D});
    const dealii::hp::QCollection<1>   nm_surface_quad_1D(dealii::QGauss<1>{nm_n_q_points_1D});
    const dealii::hp::MappingCollection<dim> nm_mapping_collection(mapping);
    const dealii::hp::FECollection<dim>      nm_fe_collection(dof_handler.get_fe());

    dealii::LinearAlgebra::distributed::Vector<number> nm_locally_relevant_level_set_vector;

    std::unique_ptr<dealii::NonMatching::MeshClassifier<dim>> nm_mesh_classifier;
    std::unique_ptr<dealii::NonMatching::FEValues<dim>>       nm_non_matching_fe_values;

    // only for dim == 1
    std::map<number, number> total_nodal_weights; // TODO: also necessary for other dimensions?

    if (use_mca == false) // this is an expensive step; execute only if needed
      {
        nm_mesh_classifier = std::make_unique<dealii::NonMatching::MeshClassifier<dim>>(
          dof_handler, nm_locally_relevant_level_set_vector);

        dealii::IndexSet locally_relevant_dofs;
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        nm_locally_relevant_level_set_vector.reinit(dof_handler.locally_owned_dofs(),
                                                    locally_relevant_dofs,
                                                    level_set_vector.get_mpi_communicator());
        nm_locally_relevant_level_set_vector.copy_locally_owned_data_from(level_set_vector);
        nm_locally_relevant_level_set_vector.update_ghost_values();
        nm_mesh_classifier->reclassify();

        dealii::NonMatching::RegionUpdateFlags nm_region_update_flags;
        nm_region_update_flags.surface =
          dealii::update_quadrature_points | dealii::update_JxW_values;

        nm_non_matching_fe_values = std::make_unique<dealii::NonMatching::FEValues<dim>>(
          nm_mapping_collection,
          nm_fe_collection,
          nm_quad,
          nm_surface_quad_1D,
          nm_region_update_flags,
          *nm_mesh_classifier,
          dof_handler,
          nm_locally_relevant_level_set_vector);
      }

    for (const auto &cell : dof_handler.active_cell_iterators_on_level(
           dof_handler.get_triangulation().n_global_levels() - 1))
      {
        if (cell->is_locally_owned())
          {
            // determine if cell is cut by the interface and if yes, determine the quadrature
            // point location (at the reference cell) and weight
            const auto fu_mca = [&]() -> std::tuple<std::vector<dealii::Point<dim>>,
                                                    std::vector<dealii::Point<dim>>,
                                                    std::vector<number>> {
              // determine points and cells of aux surface triangulation
              std::vector<dealii::Point<dim>>                       surface_vertices;
              std::vector<dealii::CellData<dim == 1 ? 1 : dim - 1>> surface_cells;

              // run marching cube algorithm
              if (dim > 1)
                mc.process_cell(
                  cell, level_set_vector, contour_value, surface_vertices, surface_cells);
              else
                mc.process_cell(cell, level_set_vector, contour_value, surface_vertices);

              if (surface_vertices.size() == 0)
                return {}; // cell is not cut by interface -> no quadrature points have the be
                           // determined

              std::vector<dealii::Point<dim>> points_real;
              std::vector<dealii::Point<dim>> points;
              std::vector<number>             weights;

              if constexpr (dim == 1)
                {
                  AssertThrow(surface_vertices.size() == 1,
                              dealii::ExcMessage("The MCA in 1D found too many points."));

                  points_real.emplace_back(surface_vertices[0]);
                  points.emplace_back(mapping.transform_real_to_unit_cell(cell, points_real[0]));

                  const auto p = points_real[0][0];

                  // check if point is vertex
                  if (points_real[0].distance(cell->vertex(0)) < 1e-16 ||
                      points_real[0].distance(cell->vertex(cell->n_vertices() - 1)) < 1e-16)
                    {
                      // consistency check, if MCA returns 2 cells if a corner node is found
                      if (total_nodal_weights.find(p) == total_nodal_weights.end())
                        total_nodal_weights[p] = 0.5;
                      else
                        total_nodal_weights[p] += 0.5;

                      weights.emplace_back(0.5);
                    }
                  else
                    {
                      if (total_nodal_weights.find(p) == total_nodal_weights.end())
                        total_nodal_weights[p] = 1.0;
                      else
                        AssertThrow(false,
                                    dealii::ExcMessage("Inconsistency found with MCA. Abort..."));

                      weights.emplace_back(1.);
                    }
                }
              else
                {
                  // create aux triangulation of subcells
                  dealii::Triangulation<dim == 1 ? 1 : dim - 1, dim> surface_triangulation;
                  surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

                  // for 3D MCA
                  dealii::FE_SimplexP<dim == 1 ? 1 : dim - 1, dim> fe_simplex(
                    dof_handler.get_fe().tensor_degree());
                  dealii::MappingFE<dim == 1 ? 1 : dim - 1, dim> simplex_mapping(fe_simplex);

                  dealii::FE_Nothing<dim == 1 ? 1 : dim - 1, dim> fe;
                  dealii::FEValues<dim == 1 ? 1 : dim - 1, dim>   fe_eval(
                    simplex_mapping,
                    fe,
                    surface_quad,
                    dealii::update_quadrature_points | dealii::update_JxW_values);

                  // loop over all cells ...
                  for (const auto &sub_cell : surface_triangulation.active_cell_iterators())
                    {
                      fe_eval.reinit(sub_cell);

                      // ... and collect quadrature points and weights
                      for (const auto &q : fe_eval.quadrature_point_indices())
                        {
                          points_real.emplace_back(fe_eval.quadrature_point(q));
                          points.emplace_back(
                            mapping.transform_real_to_unit_cell(cell, fe_eval.quadrature_point(q)));

                          weights.emplace_back(fe_eval.JxW(q));
                        }
                    }
                }
              return {points_real, points, weights};
            };

            const auto fu_non_matching = [&]() -> std::tuple<std::vector<dealii::Point<dim>>,
                                                             std::vector<dealii::Point<dim>>,
                                                             std::vector<number>> {
              nm_non_matching_fe_values->reinit(cell);

              const std::optional<dealii::NonMatching::FEImmersedSurfaceValues<dim>>
                &surface_fe_values = nm_non_matching_fe_values->get_surface_fe_values();

              if (surface_fe_values.has_value() == false)
                return {};

              std::vector<dealii::Point<dim>> points;

              for (const auto &q : surface_fe_values->get_quadrature_points())
                points.emplace_back(mapping.transform_real_to_unit_cell(cell, q));

              return {surface_fe_values->get_quadrature_points(),
                      points,
                      surface_fe_values->get_JxW_values()};
            };

            const auto [points_real, points, weights] = use_mca ? fu_mca() : fu_non_matching();

            if (points_real.size() == 0)
              continue; // cell is not cut but the interface -> nothing to do

            evaluate_at_interface_points(cell, points_real, points, weights);
          }
      }

    // check if total nodal weights are equal to 1
    if (dim == 1)
      for (const auto &[k, w] : total_nodal_weights)
        AssertThrow(std::abs(w - 1) < 1e-10,
                    dealii::ExcMessage(
                      "The MCA delivered only one cell corresponding to the corner node " +
                      std::to_string(k) + ". The total nodal weight is " + std::to_string(w) +
                      ". Abort..."));

    if (not is_ghosted)
      level_set_vector.zero_out_ghost_values();
  }

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
                             const bool         use_mca        = true)
  {
    std::vector<std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                           std::vector<dealii::Point<dim>> /*quad_points*/,
                           std::vector<number> /*weights*/
                           >>
      surface_mesh_info;

    evaluate_at_interface<dim, number>(
      dof_handler,
      mapping,
      level_set,
      [&](const auto &cell, const auto &, const auto &unit_points, const auto &weights) {
        if (unit_points.size() > 0)
          surface_mesh_info.emplace_back(cell, unit_points, weights);
      },
      contour_value,
      n_subdivisions,
      tolerance,
      use_mca);

    return surface_mesh_info;
  }

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
    const unsigned int                                             n_subdivisions_MCA = 1)
  {
    const bool level_set_is_ghosted = level_set_vector.has_ghost_elements();
    if (not level_set_is_ghosted)
      level_set_vector.update_ghost_values();

    const bool normal_vector_is_ghosted = normal_vector.has_ghost_elements();
    if (not normal_vector_is_ghosted)
      normal_vector.update_ghost_values();

    dealii::FEPointEvaluation<dim, dim> phi_normal(mapping, fe_normal, dealii::update_values);

    std::vector<number>                          buffer;
    std::vector<number>                          buffer_dim;
    std::vector<dealii::types::global_dof_index> local_dof_indices;

    global_points_normal_to_interface.clear();
    global_points_normal_to_interface_pointer.clear();

    global_points_normal_to_interface_pointer = {0};

    const auto bounding_box =
      dealii::GridTools::compute_bounding_box(dof_handler_ls.get_triangulation());
    const auto boundary_points = bounding_box.get_boundary_points();


    const auto collect_points_along_normal = [&](const auto                  &cell,
                                                 const auto                  &real_points,
                                                 const auto                  &unit_points,
                                                 [[maybe_unused]] const auto &weights) {
      local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      buffer.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(local_dof_indices);

      phi_normal.reinit(cell, unit_points);
      /*
       * gather_evaluate normal for the points at the interface
       */
      {
        buffer_dim.resize(fe_normal.n_dofs_per_cell());
        for (int i = 0; i < dim; ++i)
          {
            cell->get_dof_values(normal_vector.block(i), buffer.begin(), buffer.end());

            for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
              buffer_dim[fe_normal.component_to_system_index(i, c)] = buffer[c];
          }

        // normalize
        for (unsigned int c = 0; c < cell->get_fe().n_dofs_per_cell(); ++c)
          {
            number norm = 0.0;
            for (int i = 0; i < dim; ++i)
              norm += dealii::Utilities::fixed_power<2>(
                buffer_dim[fe_normal.component_to_system_index(i, c)]);

            norm = std::sqrt(norm);

            if (norm > 1e-16)
              {
                for (int i = 0; i < dim; ++i)
                  buffer_dim[fe_normal.component_to_system_index(i, c)] /= norm;
              }
          }

        phi_normal.evaluate(dealii::make_array_view(buffer_dim), dealii::EvaluationFlags::values);
      }

      /*
       * loop over generated points from MCA
       */
      for (unsigned int p_index = 0; p_index < real_points.size(); ++p_index)
        {
          /*
           * generate points (x) along normal
           *
           *          x
           *          |
           *          x
           *          |
           *    ---------------  ls = contour_value
           *          |
           *          x
           *          |
           *          x
           */
          std::vector<dealii::Point<dim>> points_normal_to_interface;

          const auto unit_normal = phi_normal.get_value(p_index);

          UtilityFunctions::generate_points_along_vector<dim>(points_normal_to_interface,
                                                              real_points[p_index],
                                                              unit_normal,
                                                              max_distance_per_side,
                                                              n_inc_per_side,
                                                              bidirectional);

          // check if point is outside domain and if so then project it back to
          // the domain
          //
          // @todo: compute intersection between normal and bounding box and use
          // this point
          for (unsigned int d = 0; d < dim; ++d)
            {
              for (unsigned int counter = 0; counter < points_normal_to_interface.size(); counter++)
                {
                  if (points_normal_to_interface[counter][d] < boundary_points.first[d])
                    {
                      points_normal_to_interface.erase(points_normal_to_interface.begin() +
                                                       counter);
                      continue;
                    }
                  else if (points_normal_to_interface[counter][d] > boundary_points.second[d])
                    points_normal_to_interface.erase(points_normal_to_interface.begin() + counter);
                }
            }

          global_points_normal_to_interface.insert(global_points_normal_to_interface.end(),
                                                   points_normal_to_interface.begin(),
                                                   points_normal_to_interface.end());
          global_points_normal_to_interface_pointer.emplace_back(
            global_points_normal_to_interface.size());
        }
    };

    evaluate_at_interface<dim, number>(dof_handler_ls,
                                       mapping,
                                       level_set_vector,
                                       collect_points_along_normal,
                                       contour_value, /*contour value*/
                                       n_subdivisions_MCA /*n_subdivisions*/);

    if (not level_set_is_ghosted)
      level_set_vector.zero_out_ghost_values();
    if (not normal_vector_is_ghosted)
      normal_vector.zero_out_ghost_values();
  }

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
    const number                                              lower_threshold = 0.5)
  {
    const bool has_ghost_elements = level_set_heaviside.has_ghost_elements();

    if (not has_ghost_elements)
      level_set_heaviside.update_ghost_values();

    dealii::Vector<number> hs_local(scratch_data.get_n_dofs_per_cell(ls_dof_idx));
    for (const auto &cell : scratch_data.get_dof_handler(ls_dof_idx).active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_values(level_set_heaviside, hs_local);
            const number min_ls = *std::min_element(hs_local.begin(), hs_local.end());
            cell->set_material_id(min_ls >= lower_threshold ? 1 : 0);
          }
      }

    if (not has_ghost_elements)
      level_set_heaviside.zero_out_ghost_values();

    // communicate local data to ghost cells
    if (get_triangulation_type(scratch_data.get_triangulation()) !=
        dealii::TriangulationType::serial)
      {
        using active_cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;
        auto pack                  = [](const active_cell_iterator &cell) -> unsigned int {
          return cell->material_id();
        };

        auto unpack = [](const active_cell_iterator &cell, const unsigned int material_id) -> void {
          cell->set_material_id(material_id);
        };
        if (const auto tria_shared = dynamic_cast<dealii::parallel::shared::Triangulation<dim> *>(
              const_cast<dealii::Triangulation<dim> *>(&scratch_data.get_triangulation())))
          {
            AssertThrow(
              tria_shared->with_artificial_cells(),
              dealii::ExcMessage(
                "For a shared triangulation, make sure that you allow to create artificial cells."
                "Otherwise every cell is a ghost cell, which prohibits the use of this functionality."));
          }

        dealii::GridTools::exchange_cell_data_to_ghosts<unsigned int, dealii::Triangulation<dim>>(
          scratch_data.get_triangulation(), pack, unpack);
      }
  }
} // namespace MeltPoolDG::LevelSet::Tools
