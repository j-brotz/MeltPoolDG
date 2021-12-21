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

} // namespace MeltPoolDG::LevelSet::Tools
