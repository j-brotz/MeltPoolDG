/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, December 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::LevelSet::Tools
{
  using namespace dealii;

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
   * This utility function computes the projection of nodal points from a given @p dof_handler_req
   * within the interfacial region (0 < level_set_as_heaviside < 1) on the interface, the latter
   * represented by @p level_set_as_heaviside = 0. To this end, by means of the @p distance vector,
   * holding the distance to the interface, for a given point an iterative procedure computes the
   * projected point (adjustable by @p max_iterations), ideally characterized by zero
   * distance. The result consists of a vector of projected points @return
   * projected_points_at_interface and of the corresponding @return dof_indices according to the
   * dof_handler_requested.
   */
  template <int dim>
  std::pair<std::vector<types::global_dof_index>, std::vector<Point<dim>>>
  compute_projected_points_at_interface(
    const Mapping<dim> &                             mapping,
    const DoFHandler<dim> &                          dof_handler_ls,
    const DoFHandler<dim> &                          dof_handler_req,
    const VectorType &                               level_set_as_heaviside,
    const VectorType &                               distance,
    const BlockVectorType &                          normal_vector,
    Utilities::MPI::RemotePointEvaluation<dim, dim> &remote_point_evaluation =
      Utilities::MPI::RemotePointEvaluation<dim, dim>(1e-6 /*tolerance*/, true /*unique mapping*/),
    const unsigned int max_iterations   = 5,
    const double       rel_tol_distance = 1e-5)
  {
    /*
     * tolerance to be reached for the distance to the level set = 0 isosurface
     */
    const double tol_distance = distance.linfty_norm() * rel_tol_distance;

    /*
     * read MPI communicator
     */
    const auto mpi_comm = dof_handler_ls.get_communicator();
    /*
     * collect evaluation points
     */
    distance.update_ghost_values();
    normal_vector.update_ghost_values();
    level_set_as_heaviside.update_ghost_values();

    FEValues<dim> ls_values(mapping,
                            dof_handler_ls.get_fe(),
                            Quadrature<dim>(
                              dof_handler_req.get_fe().base_element(0).get_unit_support_points()),
                            update_values);

    FEValues<dim> req_values(mapping,
                             dof_handler_req.get_fe(),
                             Quadrature<dim>(
                               dof_handler_req.get_fe().base_element(0).get_unit_support_points()),
                             update_quadrature_points);

    /*
     * vectors to be filled: projected points to the interface corresponding to DoF indices
     */
    std::vector<Point<dim>>              projected_points_at_interface;
    std::vector<types::global_dof_index> dof_indices;

    /*
     * temporary values at cell nodes
     */
    const unsigned int                   n_q_points = ls_values.get_quadrature().size();
    std::vector<double>                  hs_temp(n_q_points);
    std::vector<types::global_dof_index> temp_local_dof_indices(n_q_points);

    const auto bounding_box = GridTools::compute_bounding_box(dof_handler_ls.get_triangulation());
    const auto boundary_points = bounding_box.get_boundary_points();

    /*
     * fill initial evaluation points with node coordinates
     */
    typename DoFHandler<dim>::active_cell_iterator req_cell = dof_handler_req.begin_active();

    for (const auto &cell : dof_handler_ls.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            ls_values.reinit(cell);
            ls_values.get_function_values(level_set_as_heaviside, hs_temp);

            req_values.reinit(req_cell);
            req_cell->get_dof_indices(temp_local_dof_indices);

            for (const auto q : req_values.quadrature_point_indices())
              {
                // consider only points in narrow band
                if (hs_temp[q] < 1.0 && hs_temp[q] > 0.0)
                  {
                    projected_points_at_interface.push_back(req_values.quadrature_point(q));
                    dof_indices.push_back(temp_local_dof_indices[q]);
                  }
              }
          }
        ++req_cell;
      }

    /*
     * Update evaluation points max_iterations times by iteratively projecting
     * the point to the interface.
     */
    const double tolerance_normal_vector =
      UtilityFunctions::compute_numerical_zero_of_norm<dim>(dof_handler_ls.get_triangulation(),
                                                            mapping);

    /*
     * points that are not (yet) at the interface and still needs to be processed
     */
    std::vector<Point<dim>>   processed_points = projected_points_at_interface;
    std::vector<unsigned int> processed_points_idx(processed_points.size());
    std::iota(processed_points_idx.begin(), processed_points_idx.end(), 0);

    int                 n_processed_points = 0;
    std::vector<double> evaluation_values_distance;

    for (unsigned int j = 0; j < max_iterations; ++j)
      {
        remote_point_evaluation.reinit(processed_points,
                                       dof_handler_req.get_triangulation(),
                                       mapping);

        evaluation_values_distance =
          dealii::VectorTools::point_values<1>(remote_point_evaluation, dof_handler_ls, distance);

        std::array<std::vector<double>, dim> evaluation_values_normal;

        for (int comp = 0; comp < dim; ++comp)
          evaluation_values_normal[comp] =
            dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                 dof_handler_ls,
                                                 normal_vector.block(comp));

        std::vector<Point<dim>>   processed_points_new;
        std::vector<unsigned int> processed_points_idx_new;

        for (unsigned int counter = 0; counter < processed_points.size(); ++counter)
          {
            /*
             * skip point where the distance to the interface is already close enough
             */
            if (std::abs(evaluation_values_distance[counter]) <= tol_distance)
              continue;
            /*
             * compute unit normal vector
             */
            Point<dim> unit_normal;
            for (unsigned int comp = 0; comp < dim; ++comp)
              unit_normal[comp] = evaluation_values_normal[comp][counter];

            const auto n_norm = unit_normal.norm();
            unit_normal = (n_norm > tolerance_normal_vector) ? unit_normal / n_norm : Point<dim>();

            /*
             * compute corresponding point at level set == 0.0
             */
            for (unsigned int d = 0; d < dim; ++d)
              {
                processed_points[counter][d] -=
                  evaluation_values_distance[counter] * unit_normal[d];

                // check if point is outside domain and if so then project it back to
                // the domain
                if (processed_points[counter][d] < boundary_points.first[d])
                  processed_points[counter][d] = boundary_points.first[d];
                else if (processed_points[counter][d] > boundary_points.second[d])
                  processed_points[counter][d] = boundary_points.second[d];
              }

            projected_points_at_interface[processed_points_idx[counter]] =
              processed_points[counter];
            processed_points_new.emplace_back(processed_points[counter]);
            processed_points_idx_new.emplace_back(processed_points_idx[counter]);
          }

        /*
         * remove points from processing that are already at the interface
         */
        processed_points     = processed_points_new;
        processed_points_idx = processed_points_idx_new;

        /*
         * If every point is close enough to the interface, we are finished.
         */
        n_processed_points = processed_points.size();
        n_processed_points = Utilities::MPI::sum(n_processed_points, mpi_comm);

        if (n_processed_points == 0)
          break;
      }
    /*
     * compute maximum distance of projected points to the level set 0 isosurface
     */
    double max_distance =
      evaluation_values_distance.size() == 0 ?
        0.0 :
        *std::max_element(evaluation_values_distance.begin(), evaluation_values_distance.end());

    max_distance = Utilities::MPI::max(max_distance, mpi_comm);

    if (n_processed_points > 0 && max_distance > tol_distance)
      {
        dealii::ConditionalOStream pcout(std::cout,
                                         Utilities::MPI::this_mpi_process(mpi_comm) == 0);
        pcout << "WARNING: The tolerance of " << processed_points.size()
              << " points is not yet attained. Max distance value: " << max_distance << std::endl;
      }

    distance.zero_out_ghost_values();
    normal_vector.zero_out_ghost_values();
    level_set_as_heaviside.zero_out_ghost_values();
    /*
     * debug
     */
    if (false)
      {
        const auto global_points_normal_to_interface_all =
          Utilities::MPI::reduce<std::vector<Point<dim>>>(
            projected_points_at_interface, mpi_comm, [](const auto &a, const auto &b) {
              auto result = a;
              result.insert(result.end(), b.begin(), b.end());
              return result;
            });

        std::ofstream myfile;
        myfile.open("generated_points.dat");

        for (const auto &p : global_points_normal_to_interface_all)
          {
            for (unsigned int d = 0; d < dim; ++d)
              myfile << p[d] << " ";
            myfile << std::endl;
          }

        myfile.close();
      }

    return {dof_indices, projected_points_at_interface};
  }

  template <int dim, int n_components = 1>
  void
  broadcast_interface_value_to_vector(
    const Mapping<dim> &                             mapping,
    const DoFHandler<dim> &                          dof_handler_ls,
    const DoFHandler<dim> &                          dof_handler_req,
    const VectorType &                               level_set_as_heaviside,
    const VectorType &                               distance,
    const BlockVectorType &                          normal_vector,
    const VectorType &                               solution_in,
    VectorType &                                     solution_out,
    Utilities::MPI::RemotePointEvaluation<dim, dim> &remote_point_evaluation =
      Utilities::MPI::RemotePointEvaluation<dim, dim>(1e-6 /*tolerance*/, true /*unique mapping*/),
    const unsigned int max_iterations   = 5,
    const double       rel_tol_distance = 1e-5)
  {
    AssertThrow(n_components == 1, ExcNotImplemented());

    const auto [dof_indices, evaluation_points] =
      compute_projected_points_at_interface<dim>(mapping,
                                                 dof_handler_ls,
                                                 dof_handler_req,
                                                 level_set_as_heaviside,
                                                 distance,
                                                 normal_vector,
                                                 remote_point_evaluation,
                                                 max_iterations,
                                                 rel_tol_distance);

    remote_point_evaluation.reinit(evaluation_points, dof_handler_req.get_triangulation(), mapping);

    solution_in.update_ghost_values();

    const auto vals =
      dealii::VectorTools::point_values<1>(remote_point_evaluation, dof_handler_req, solution_in);
    solution_in.zero_out_ghost_values();

    /*
     * compute evaporative mass flux from the temperature value at the interface
     */
    solution_out = 0.0;

    for (unsigned int i = 0; i < evaluation_points.size(); ++i)
      solution_out[dof_indices[i]] = vals[i];
  }


  /**
   * For two indicator vectors, representing e.g. implicit geometries, this function computes a
   * boolean operation and returns the resulting vector. The user has to take care on distributing
   * relevant constraints afterwards.
   */
  template <typename VectorType>
  VectorType
  merge_two_indicator_fields(const VectorType &indicator_1,
                             const VectorType &indicator_2,
                             BooleanType       type                     = BooleanType::Union,
                             const double      indicator_value_interior = 1.0,
                             const double      indicator_value_exterior = -1.0)
  {
    AssertThrow(indicator_1.size() == indicator_2.size(),
                ExcMessage("The two level set vectors to be merged must be of equal length."));

    AssertThrow(indicator_value_interior != indicator_value_exterior,
                ExcMessage("The indicator value in the interior of the implicit geometry must be "
                           "different than the one in the exterior. Make sure that the function "
                           "arguments indicator_value_interior and indicator_value_exterior are "
                           "set correctly."));

    AssertThrow(indicator_1.linfty_norm() ==
                  std::max(indicator_value_exterior, indicator_value_interior),
                ExcMessage(
                  "The maximum indicator value does not correspond to the given bounds "
                  "of indicator_value_interior and indicator_value_exterior. Make sure that "
                  "the indicator_value_interior and indicator_value_exterior comply with the "
                  "indicator field."));

    VectorType merge_indicator(indicator_1);

    bool is_interior_larger_than_exterior = indicator_value_interior > indicator_value_exterior;

    for (const auto &i : indicator_1.locally_owned_elements())
      switch (type)
        {
          case BooleanType::Union:
            merge_indicator[i] = (is_interior_larger_than_exterior) ?
                                   std::max(indicator_1[i], indicator_2[i]) :
                                   std::min(indicator_1[i], indicator_2[i]);
            break;
          case BooleanType::Intersection:
            merge_indicator[i] = (is_interior_larger_than_exterior) ?
                                   std::min(indicator_1[i], indicator_2[i]) :
                                   std::max(indicator_1[i], indicator_2[i]);
            break;
          case BooleanType::Subtraction:
            merge_indicator[i] =
              (indicator_1[i] == indicator_2[i] && indicator_1[i] == indicator_value_interior) ?
                indicator_value_exterior :
                indicator_1[i];
            break;
          default:
            AssertThrow(false, ExcNotImplemented());
        }

    merge_indicator.compress(VectorOperation::insert);

    return merge_indicator;
  }

  /**
   * This utility function enables the evaluation of variables at interfaces defined
   * implicitly by a @p level_set_vector at a certain @p contour_value. The marching
   * cube algorithm is used to determine @p points (unit points at the reference cell)
   * and corresponding integration @p weights at the interface. The lambda function
   * @p evaluate_at_interface_points is called for every active cell and can be used
   * to e.g. fill a DoF vector.
   */
  template <int dim>
  void
  evaluate_at_interface(
    const DoFHandler<dim> &                                         dof_handler,
    const Mapping<dim> &                                            mapping,
    const VectorType &                                              level_set_vector,
    const std::function<void(const typename DoFHandler<dim>::active_cell_iterator & /*cell*/,
                             const std::vector<Point<dim>> & /*points_real*/,
                             const std::vector<Point<dim>> & /*points_reference*/,
                             const std::vector<double> & /*JxW*/)> &evaluate_at_interface_points,
    const double                                                    contour_value  = 0.0,
    const unsigned int                                              n_subdivisions = 1,
    const bool                                                      use_mca        = true)
  {
    AssertThrow(dim > 1, ExcNotImplemented());

    // data structures for marching-cube algorithm
    const QGauss<dim - 1> surface_quad(dof_handler.get_fe().degree + 1);

    GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(mapping,
                                                         dof_handler.get_fe(), // todo
                                                         n_subdivisions);

    const bool is_ghosted = level_set_vector.has_ghost_elements();

    if (!is_ghosted)
      level_set_vector.update_ghost_values();

    // data structures for NonMatching::FEValues
    const unsigned int nm_n_q_points_1D =
      std::max<unsigned int>(dof_handler.get_fe().degree, n_subdivisions + 1);
    const hp::QCollection<dim>       nm_quad(QGauss<dim>{nm_n_q_points_1D});
    const hp::QCollection<1>         nm_surface_quad_1D(QGauss<1>{nm_n_q_points_1D});
    const hp::MappingCollection<dim> nm_mapping_collection(mapping);
    const hp::FECollection<dim>      nm_fe_collection(dof_handler.get_fe());

    VectorType nm_locally_relevant_level_set_vector;

    std::unique_ptr<NonMatching::MeshClassifier<dim>> nm_mesh_classifier;
    std::unique_ptr<NonMatching::FEValues<dim>>       nm_non_matching_fe_values;

    if (use_mca == false) // this is an expensive step; execute only if needed
      {
        nm_mesh_classifier =
          std::make_unique<NonMatching::MeshClassifier<dim>>(dof_handler,
                                                             nm_locally_relevant_level_set_vector);

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        nm_locally_relevant_level_set_vector.reinit(dof_handler.locally_owned_dofs(),
                                                    locally_relevant_dofs,
                                                    level_set_vector.get_mpi_communicator());
        nm_locally_relevant_level_set_vector.copy_locally_owned_data_from(level_set_vector);
        nm_locally_relevant_level_set_vector.update_ghost_values();
        nm_mesh_classifier->reclassify();

        NonMatching::RegionUpdateFlags nm_region_update_flags;
        nm_region_update_flags.surface = update_quadrature_points | update_JxW_values;

        nm_non_matching_fe_values =
          std::make_unique<NonMatching::FEValues<dim>>(nm_mapping_collection,
                                                       nm_fe_collection,
                                                       nm_quad,
                                                       nm_surface_quad_1D,
                                                       nm_region_update_flags,
                                                       *nm_mesh_classifier,
                                                       dof_handler,
                                                       nm_locally_relevant_level_set_vector);
      }

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            // determine if cell is cut by the interface and if yes, determine the quadrature
            // point location (at the reference cell) and weight
            const auto fu_mca = [&]()
              -> std::tuple<std::vector<Point<dim>>, std::vector<Point<dim>>, std::vector<double>> {
              // determine points and cells of aux surface triangulation
              std::vector<Point<dim>>        surface_vertices;
              std::vector<CellData<dim - 1>> surface_cells;

              // run marching cube algorithm
              mc.process_cell(
                cell, level_set_vector, contour_value, surface_vertices, surface_cells);

              if (surface_vertices.size() == 0)
                return {}; // cell is not cut by interface -> no quadrature points have the be
                           // determined

              std::vector<Point<dim>> points_real;
              std::vector<Point<dim>> points;
              std::vector<double>     weights;

              // create aux triangulation of subcells
              Triangulation<dim - 1, dim> surface_triangulation;
              surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

              FE_Nothing<dim - 1, dim> fe;
              FEValues<dim - 1, dim>   fe_eval(fe,
                                             surface_quad,
                                             update_quadrature_points | update_JxW_values);

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
              return {points_real, points, weights};
            };

            const auto fu_non_matching = [&]()
              -> std::tuple<std::vector<Point<dim>>, std::vector<Point<dim>>, std::vector<double>> {
              nm_non_matching_fe_values->reinit(cell);

              const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
                &surface_fe_values = nm_non_matching_fe_values->get_surface_fe_values();

              if (surface_fe_values.has_value() == false)
                return {};

              std::vector<Point<dim>> points;

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
    if (!is_ghosted)
      level_set_vector.zero_out_ghost_values();
  }

  template <int dim>
  std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                         std::vector<Point<dim>> /*quad_points*/,
                         std::vector<double> /*weights*/
                         >>
  generate_surface_mesh_info(const DoFHandler<dim> &dof_handler,
                             const Mapping<dim> &   mapping,
                             const VectorType &     level_set_as_heaviside,
                             const double           contour_value  = 0.0,
                             const unsigned int     n_subdivisions = 1,
                             const bool             use_mca        = true)
  {
    std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                           std::vector<Point<dim>> /*quad_points*/,
                           std::vector<double> /*weights*/
                           >>
      surface_mesh_info;

    if constexpr (dim > 1)
      {
        evaluate_at_interface<dim>(
          dof_handler,
          mapping,
          level_set_as_heaviside,
          [&](const auto &cell, const auto &, const auto &unit_points, const auto &weights) {
            if (unit_points.size() > 0)
              surface_mesh_info.emplace_back(cell, unit_points, weights);
          },
          contour_value,
          n_subdivisions,
          use_mca);
      }
    else
      {
        (void)dof_handler;
        (void)mapping;
        (void)level_set_as_heaviside;
        (void)contour_value;
        (void)n_subdivisions;
        (void)use_mca;

        AssertThrow(false, ExcNotImplemented());
      }

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
  template <int dim>
  void
  generate_points_along_normal(std::vector<Point<dim>> &  global_points_normal_to_interface,
                               std::vector<unsigned int> &global_points_normal_to_interface_pointer,
                               const DoFHandler<dim> &    dof_handler_ls,
                               const FESystem<dim> &      fe_normal,
                               const Mapping<dim> &       mapping,
                               const VectorType &         level_set_vector,
                               const BlockVectorType &    normal_vector,
                               const double               max_distance_per_side,
                               const unsigned int         n_inc_per_side,
                               const bool                 bidirectional      = true,
                               const double               contour_value      = 0.0,
                               const unsigned int         n_subdivisions_MCA = 1)
  {
    level_set_vector.update_ghost_values();
    normal_vector.update_ghost_values();

    FEPointEvaluation<dim, dim> phi_normal(mapping, fe_normal, update_values);

    std::vector<double>                  buffer;
    std::vector<double>                  buffer_dim;
    std::vector<types::global_dof_index> local_dof_indices;

    global_points_normal_to_interface.clear();
    global_points_normal_to_interface_pointer.clear();

    global_points_normal_to_interface_pointer = {0};

    const auto bounding_box = GridTools::compute_bounding_box(dof_handler_ls.get_triangulation());
    const auto boundary_points = bounding_box.get_boundary_points();


    const auto collect_points_along_normal = [&](const auto &                 cell,
                                                 const auto &                 real_points,
                                                 const auto &                 unit_points,
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
            double norm = 0.0;
            for (int i = 0; i < dim; ++i)
              norm += std::pow(buffer_dim[fe_normal.component_to_system_index(i, c)], 2);

            norm = std::sqrt(norm);
            for (int i = 0; i < dim; ++i)
              buffer_dim[fe_normal.component_to_system_index(i, c)] /= norm;
          }

        phi_normal.evaluate(make_array_view(buffer_dim), EvaluationFlags::values);
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
          std::vector<Point<dim>> points_normal_to_interface;

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

    evaluate_at_interface<dim>(dof_handler_ls,
                               mapping,
                               level_set_vector,
                               collect_points_along_normal,
                               contour_value, /*contour value*/
                               n_subdivisions_MCA /*n_subdivisions*/);

    level_set_vector.zero_out_ghost_values();
    normal_vector.zero_out_ghost_values();
  }

} // namespace MeltPoolDG::LevelSet::Tools
