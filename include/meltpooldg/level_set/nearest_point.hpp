#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/rtree.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <meltpooldg/level_set/nearest_point_data.hpp>
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
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
                 std::optional<std::reference_wrapper<dealii::TimerOutput>> timer_output = {})
      : mapping(mapping)
      , dof_handler_ls(dof_handler_signed_distance)
      , signed_distance(signed_distance)
      , normal_vector(normal_vector)
      , additional_data(additional_data)
      , remote_point_evaluation(remote_point_evaluation)
      , tol_distance(additional_data.rel_tol *
                     dealii::GridTools::minimal_cell_diameter(dof_handler_ls.get_triangulation()) /
                     std::sqrt(dim))
      , narrow_band_threshold(additional_data.narrow_band_threshold > 0 ?
                                additional_data.narrow_band_threshold :
                                signed_distance.linfty_norm() * 0.9999)
      , tolerance_normal_vector(
          UtilityFunctions::compute_numerical_zero_of_norm<dim>(dof_handler_ls.get_triangulation(),
                                                                mapping))
      , mpi_comm(dof_handler_ls.get_communicator())
      , pcout(std::cout,
              dealii::Utilities::MPI::this_mpi_process(
                dof_handler_signed_distance.get_communicator()) == 0)
      , timer_output(timer_output)
    {
      if (additional_data.verbosity_level > 0)
        {
          Journal::print_line(pcout,
                              "narrow band threshold " + std::to_string(narrow_band_threshold) +
                                "    isocontour = " + std::to_string(additional_data.isocontour),
                              "nearest_point");
        }
      AssertThrow(dim <= 2 or
                    additional_data.type != NearestPointType::closest_point_normal_collinear,
                  dealii::ExcNotImplemented());

      AssertThrow(narrow_band_threshold > 0,
                  dealii::ExcMessage("Narrow band threshold must be larger than zero."));
    }

    /**
     * Update the nearest points of the nodal points from a given DoFHandler @p dof_handler_req.
     */
    void
    reinit(const dealii::DoFHandler<dim> &dof_handler_req)
    {
      std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
      if (timer_output)
        timer_scope =
          std::make_unique<dealii::TimerOutput::Scope>(timer_output.value(),
                                                       ScopedName("nearest_point::reinit"));

      std::unique_ptr<dealii::TimerOutput::Scope> timer_scope_local;
      if (timer_output)
        timer_scope_local = std::make_unique<dealii::TimerOutput::Scope>(
          timer_output.value(), ScopedName("nearest_point::reinit::collect_support_points"));

      is_reinit_called = true;
      // calculate point cloud corresponding to nodes of the requested DoFHandler
      // located within the narrow band region
      const unsigned int n_components = dof_handler_req.get_fe().n_components();

      const bool signed_distance_update_ghosts = not signed_distance.has_ghost_elements();
      if (signed_distance_update_ghosts)
        signed_distance.update_ghost_values();

      const bool normal_vector_update_ghosts = not normal_vector.has_ghost_elements();
      if (normal_vector_update_ghosts)
        normal_vector.update_ghost_values();

      dealii::FEValues<dim> distance_values(
        mapping,
        dof_handler_ls.get_fe(),
        dealii::Quadrature<dim>(dof_handler_req.get_fe().base_element(0).get_unit_support_points()),
        dealii::update_values);

      dealii::FEValues<dim> req_values(
        mapping,
        dof_handler_req.get_fe(),
        dealii::Quadrature<dim>(dof_handler_req.get_fe().base_element(0).get_unit_support_points()),
        dealii::update_quadrature_points);

      // fill initial evaluation points with node coordinates (stencil)
      const unsigned int  n_q_points = distance_values.get_quadrature().size();
      std::vector<number> temp_distance(n_q_points);
      std::vector<dealii::types::global_dof_index> temp_local_dof_indices(n_q_points *
                                                                          n_components);

      typename dealii::DoFHandler<dim>::active_cell_iterator req_cell =
        dof_handler_req.begin_active();

      // free caches of stencil points, corresponding nearest points and dof indices
      stencil.clear();
      dof_indices.clear();
      projected_points_at_interface.clear();

      for (const auto &cell : dof_handler_ls.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              distance_values.reinit(cell);
              distance_values.get_function_values(signed_distance, temp_distance);

              req_values.reinit(req_cell);
              req_cell->get_dof_indices(temp_local_dof_indices);

              for (const auto q : req_values.quadrature_point_indices())
                {
                  // copied from dealii::GridTools and modified
                  //
                  // check if the rank of this process is the lowest of all cells
                  // if not, the other process will handle this cell and we don't
                  // have to do here anything in the case of unique mapping
                  unsigned int lowest_rank = numbers::invalid_unsigned_int;

                  const auto active_cells_around_point =
                    dealii::GridTools::find_all_active_cells_around_point<dim>(
                      mapping,
                      dof_handler_ls.get_triangulation(),
                      req_values.quadrature_point(q),
                      1e-6 /*tolerance*/,
                      {cell,
                       mapping.transform_real_to_unit_cell(cell, req_values.quadrature_point(q))});

                  for (const auto &cell : active_cells_around_point)
                    lowest_rank = std::min(lowest_rank, cell.first->subdomain_id());

                  if (lowest_rank != dealii::Utilities::MPI::this_mpi_process(mpi_comm))
                    continue;
                  // end of copy from dealii::GridTools

                  // early return if point was already collected in stencil
                  if (std::find(stencil.begin(), stencil.end(), req_values.quadrature_point(q)) !=
                      stencil.end())
                    continue;

                  std::vector<dealii::types::global_dof_index> dofs_at_q;

                  // collect points of narrow band
                  if (std::abs(temp_distance[q]) < narrow_band_threshold)
                    {
                      stencil.emplace_back(req_values.quadrature_point(q));

                      // create a list of component-wise dof indices
                      for (unsigned int c = 0; c < n_components; ++c)
                        dofs_at_q.emplace_back(
                          temp_local_dof_indices[dof_handler_req.get_fe().component_to_system_index(
                            c, q)]);

                      dof_indices.emplace_back(dofs_at_q);
                    }
                }
            }
          ++req_cell;
        }

      // set initial guess for closest point projection
      projected_points_at_interface = stencil;

      Assert(dealii::Utilities::MPI::sum(stencil.size(), mpi_comm) > 0,
             dealii::ExcMessage("Number of points in narrow band equal to zero."));

      if (timer_scope_local)
        timer_scope_local->stop();

      if (timer_output)
        timer_scope_local =
          std::make_unique<dealii::TimerOutput::Scope>(timer_output.value(),
                                                       ScopedName("nearest_point::reinit::search"));

      total_points_rpe = 0;
      points_not_found.clear();

      if (additional_data.type == NearestPointType::nearest_point)
        {
          local_compute_nearest_point();
        }
      else if (additional_data.type == NearestPointType::closest_point_normal)
        {
          local_compute_normal_correction(projected_points_at_interface);
        }
      else if (additional_data.type == NearestPointType::closest_point_normal_collinear)
        {
          local_compute_normal_and_tangential_correction(projected_points_at_interface);
        }
      else if (additional_data.type == NearestPointType::closest_point_normal_collinear_coquerelle)
        {
          local_compute_normal_and_tangential_correction_coquerelle(projected_points_at_interface);
        }
      else
        AssertThrow(false, dealii::ExcNotImplemented());

      if (additional_data.verbosity_level > 0 and
          additional_data.type != NearestPointType::nearest_point)
        Journal::print_line(pcout,
                            "total number of RPE points: " +
                              std::to_string(
                                dealii::Utilities::MPI::sum(total_points_rpe, mpi_comm)),
                            "nearest_point");

      if (signed_distance_update_ghosts)
        signed_distance.zero_out_ghost_values();
      if (normal_vector_update_ghosts)
        normal_vector.zero_out_ghost_values();

      AssertThrow(dof_indices.size() == projected_points_at_interface.size(),
                  dealii::ExcMessage("The size of vectors does not match."));
      if (timer_scope_local)
        timer_scope_local->stop();

      if (timer_output)
        timer_scope_local =
          std::make_unique<dealii::TimerOutput::Scope>(timer_output.value(),
                                                       ScopedName("nearest_point::reinit::rpe"));
      // TODO: where to update remote points (?)
      remote_point_evaluation.reinit(projected_points_at_interface,
                                     dof_handler_ls.get_triangulation(),
                                     mapping);
      if (timer_scope_local)
        timer_scope_local->stop();
    }

    /**
     * Getter function for the nearest points, corresponding to the nodal points of the
     * DoFHandler passed into the reinit() function.
     *
     * @note Make sure that you have called reinit() before.
     */
    const std::vector<dealii::Point<dim>> &
    get_points() const
    {
      AssertThrow(is_reinit_called, dealii::ExcMessage("You need to call reinit() first."));

      return projected_points_at_interface;
    }

    /**
     * Getter function for the DoF indices of the nearest points,  corresponding to the nodal points
     * of the DoFHandler passed into the reinit() function.
     *
     * @note Make sure that you have called reinit() before.
     */
    const std::vector<std::vector<dealii::types::global_dof_index>> &
    get_dof_indices() const
    {
      AssertThrow(is_reinit_called, dealii::ExcMessage("You need to call reinit() first."));

      return dof_indices;
    }

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
      const dealii::DoFHandler<dim>             &dof_handler_req, // TODO: remove
      const VectorType                          &solution_in,
      const bool                                 zero_out  = false,
      const std::function<number(const number)> &operation = {}) const
    {
      std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
      if (timer_output)
        timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
          timer_output.value(), ScopedName("nearest_point::fill_dof_vector"));

      AssertThrow(n_components == dof_handler_req.get_fe().n_components(),
                  dealii::ExcMessage("There is a mismatch in the number of components "
                                     "between your passed DoFHandler and the template parameter."));

      AssertThrow(is_reinit_called, dealii::ExcMessage("You need to call reinit() first."));

      const bool update_ghosts = not solution_in.has_ghost_elements();

      if (update_ghosts)
        solution_in.update_ghost_values();

      const auto vals = dealii::VectorTools::point_values<n_components>(remote_point_evaluation,
                                                                        dof_handler_req,
                                                                        solution_in);

      if (update_ghosts)
        solution_in.zero_out_ghost_values();

      // store interface values to vector
      if (zero_out)
        solution_out = 0.0;

      for (unsigned int i = 0; i < projected_points_at_interface.size(); ++i)
        {
          if constexpr (n_components > 1)
            {
              for (unsigned int d = 0; d < dof_indices[i].size(); ++d)
                if (solution_out.locally_owned_elements().is_element(dof_indices[i][d]))
                  solution_out[dof_indices[i][d]] =
                    not operation ? vals[i][d] : operation(vals[i][d]);
            }
          else if (solution_out.locally_owned_elements().is_element(dof_indices[i][0]))
            solution_out[dof_indices[i][0]] = not operation ? vals[i] : operation(vals[i]);
        }
    }


    /**
     * Write the nearest points, calculated via reinit(), to a table file @p filename.
     */
    void
    write_to_file(const std::string filename = "unmatched_points") const
    {
      const auto global_points_normal_to_interface_all =
        dealii::Utilities::MPI::reduce<std::vector<dealii::Point<dim>>>(
          projected_points_at_interface, mpi_comm, [](const auto &a, const auto &b) {
            auto result = a;
            result.insert(result.end(), b.begin(), b.end());
            return result;
          });

      if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        {
          std::ofstream myfile;
          myfile.open(filename + ".dat");

          for (const auto &p : global_points_normal_to_interface_all)
            {
              for (unsigned int d = 0; d < dim; ++d)
                myfile << p[d] << " ";
              myfile << std::endl;
            }

          myfile.close();
        }
      {
        const auto all_points_not_found =
          dealii::Utilities::MPI::reduce<std::vector<dealii::Point<dim>>>(
            points_not_found, mpi_comm, [](const auto &a, const auto &b) {
              auto result = a;
              result.insert(result.end(), b.begin(), b.end());
              return result;
            });

        if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0 and
            all_points_not_found.size() > 0)
          {
            std::ofstream myfile;
            myfile.open(filename + "_not_found.dat");

            for (const auto &p : all_points_not_found)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  myfile << p[d] << " ";

                myfile << std::endl;
              }

            myfile.close();
          }
      }
    }
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
    local_compute_normal_correction(std::vector<dealii::Point<dim>> &y)
    {
      // points that are not (yet) at the interface and still needs to be processed
      std::vector<unsigned int> unmatched_points_idx(y.size());
      std::iota(unmatched_points_idx.begin(), unmatched_points_idx.end(), 0);
      int n_unmatched_points =
        dealii::Utilities::MPI::sum(unmatched_points_idx.size(), MPI_COMM_WORLD);

      // temporary variable for signed distance
      std::vector<number> evaluation_values_distance;

      for (int j = 0; j < additional_data.max_iter; ++j)
        {
          std::vector<dealii::Point<dim>> unmatched_points(unmatched_points_idx.size());

          for (unsigned int counter = 0; counter < unmatched_points_idx.size(); ++counter)
            unmatched_points[counter] = y[unmatched_points_idx[counter]];

          std::ostringstream str;
          str << "     j=" << j << " (normal) "
              << dealii::Utilities::MPI::sum(unmatched_points_idx.size(), MPI_COMM_WORLD);

          total_points_rpe += unmatched_points.size();

          remote_point_evaluation.reinit(unmatched_points,
                                         dof_handler_ls.get_triangulation(),
                                         mapping);

          if (not remote_point_evaluation.all_points_found())
            {
              for (unsigned int i = 0; i < unmatched_points.size(); ++i)
                if (not remote_point_evaluation.point_found(i))
                  points_not_found.emplace_back(unmatched_points[i]);

              write_to_file();

              AssertThrow(false, dealii::ExcMessage("Processed point is outside domain."));
            }

          // compute signed distance at unmatched_points
          evaluation_values_distance = dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                                            dof_handler_ls,
                                                                            signed_distance);

          // shift isocontour
          if (std::abs(additional_data.isocontour >= 0)) // TODO what is the std::abs for?
            {
              for (auto &e : evaluation_values_distance)
                e -= additional_data.isocontour;
            }

          // compute unit normal
          std::vector<dealii::Point<dim>> evaluation_values_unit_normal;

          std::array<std::vector<number>, dim> evaluation_values_normal;

          for (int comp = 0; comp < dim; ++comp)
            {
              evaluation_values_normal[comp] =
                dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                     dof_handler_ls,
                                                     normal_vector.block(comp));
            }

          for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
            {
              dealii::Point<dim> unit_normal;
              for (unsigned int comp = 0; comp < dim; ++comp)
                unit_normal[comp] = evaluation_values_normal[comp][counter];

              const auto n_norm = unit_normal.norm();
              unit_normal =
                n_norm > tolerance_normal_vector ? unit_normal / n_norm : dealii::Point<dim>();

              evaluation_values_unit_normal.emplace_back(unit_normal);
            }

          std::vector<unsigned int> unmatched_points_idx_next;

          for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
            {
              // skip point where the distance to the interface is already close enough
              if (std::abs(evaluation_values_distance[counter]) <= tol_distance)
                continue;

              // compute closest point and update value in global vector
              unmatched_points[counter] -=
                evaluation_values_distance[counter] * evaluation_values_unit_normal[counter];

              y[unmatched_points_idx[counter]] = unmatched_points[counter];

              unmatched_points_idx_next.emplace_back(unmatched_points_idx[counter]);
            }

          // remove points from processing that are already at the interface
          unmatched_points_idx.swap(unmatched_points_idx_next);

          // if every point is close enough to the interface, we are finished
          n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

          str << " -> " << n_unmatched_points << " (✗)";
          if (additional_data.verbosity_level > 1)
            Journal::print_line(pcout, str.str(), "nearest_point", 2);

          if (n_unmatched_points == 0)
            break;
        }

      // compute maximum distance of projected points to the level set 0 isosurface
      number max_distance =
        evaluation_values_distance.size() == 0 ?
          0.0 :
          *std::max_element(evaluation_values_distance.begin(), evaluation_values_distance.end());

      max_distance = dealii::Utilities::MPI::max(max_distance, mpi_comm);

      if (n_unmatched_points > 0)
        {
          pcout << "WARNING: The tolerance of " << n_unmatched_points
                << " points is not yet attained. Max distance value: " << max_distance << std::endl;
          return false;
        }
      return true;
    }

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
    local_compute_normal_and_tangential_correction_coquerelle(std::vector<dealii::Point<dim>> &y)
    {
      AssertThrow(dim == 2 or dim == 3,
                  dealii::ExcMessage("Use local_compute_normal_correction for dim==1."));

      total_points_rpe = 0;

      // 0) Perform correction in normal direction
      local_compute_normal_correction(y);

      // points that are not (yet) at the interface and still need to be processed
      std::vector<unsigned int> unmatched_points_idx(y.size());
      std::iota(unmatched_points_idx.begin(), unmatched_points_idx.end(), 0);

      int n_unmatched_points =
        dealii::Utilities::MPI::sum(unmatched_points_idx.size(), MPI_COMM_WORLD);

      number max_tangential_distance = 0.0;

      // 1) Perform correction in tangential direction
      for (int it = 0; it < additional_data.max_iter and n_unmatched_points > 0; ++it)
        {
          std::vector<dealii::Point<dim>> unmatched_points(unmatched_points_idx.size());
          for (unsigned int i = 0; i < unmatched_points_idx.size(); ++i)
            unmatched_points[i] = y[unmatched_points_idx[i]];

          std::ostringstream str;
          str << " i=" << it << " (tangent) ";
          str << n_unmatched_points << " -> ";
          if (additional_data.verbosity_level > 1)
            Journal::print_line(pcout, str.str(), "nearest_point");

          // update remote point evaluation for unmatched points
          total_points_rpe += unmatched_points.size();
          remote_point_evaluation.reinit(unmatched_points,
                                         dof_handler_ls.get_triangulation(),
                                         mapping);

          // compute unit normal for each unmatched point
          std::vector<dealii::Point<dim>>              evaluation_values_unit_normal;
          std::vector<std::vector<dealii::Point<dim>>> evaluation_values_unit_tangent;

          std::array<std::vector<number>, dim> evaluation_values_normal;

          for (int comp = 0; comp < dim; ++comp)
            {
              evaluation_values_normal[comp] =
                dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                     dof_handler_ls,
                                                     normal_vector.block(comp));
            }

          for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
            {
              dealii::Point<dim> unit_normal;
              for (unsigned int comp = 0; comp < dim; ++comp)
                unit_normal[comp] = evaluation_values_normal[comp][counter];

              const auto n_norm = unit_normal.norm();
              unit_normal =
                n_norm > tolerance_normal_vector ? unit_normal / n_norm : dealii::Point<dim>();

              evaluation_values_unit_normal.emplace_back(unit_normal);

              // compute the tangent(s) for each point
              std::vector<dealii::Point<dim>> tangent;
              tangent.resize(dim - 1);

              if (n_norm > tolerance_normal_vector)
                {
                  if constexpr (dim == 2)
                    {
                      tangent[0][0] = unit_normal[1];
                      tangent[0][1] = -unit_normal[0];
                    }
                  else if constexpr (dim == 3)
                    {
                      dealii::Point<dim> temp_vec = dealii::Point<dim>::unit_vector(0);

                      // if normal vector is identical with unit vector
                      // choose different unit vector to compute the
                      // tangent
                      if ((temp_vec - unit_normal).norm() < 1e-10)
                        temp_vec = dealii::Point<dim>::unit_vector(1);

                      tangent[0] = temp_vec - (temp_vec * unit_normal) * unit_normal;
                      tangent[1] = cross_product_3d(unit_normal, tangent[0]);
                    }
                }
              else
                {
                  for (unsigned int d = 0; d < dim - 1; ++d)
                    tangent[d] = dealii::Point<dim>();
                }

              evaluation_values_unit_tangent.emplace_back(tangent);
            }

          // compute the tangential correction for each point
          std::vector<unsigned int> unmatched_points_idx_next;

          for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
            {
              // check if point needs to be corrected
              const auto distance_vec =
                unmatched_points[counter] - stencil[unmatched_points_idx[counter]];

              // determine tangential offset for each direction
              std::vector<number> omega(dim - 1);
              for (unsigned int d = 0; d < dim - 1; ++d)
                omega[d] = distance_vec * evaluation_values_unit_tangent[counter][d];

              // determine maximum tangential offset
              number max_omega = 0;
              for (unsigned int d = 0; d < dim - 1; ++d)
                max_omega = std::max(max_omega, std::abs(omega[d]));

              if (max_omega <= tol_distance)
                continue; // no need to perform a tangential corection

              max_tangential_distance = std::max(max_omega, max_tangential_distance);

              // correct point by tangential offset
              for (unsigned int d = 0; d < dim - 1; ++d)
                unmatched_points[counter] -= omega[d] * evaluation_values_unit_tangent[counter][d];

              unmatched_points_idx_next.emplace_back(unmatched_points_idx[counter]);
            }
          // 2) Proceed with the correction in normal direction
          local_compute_normal_correction(unmatched_points);

          // update points in global vector
          for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
            y[unmatched_points_idx[counter]] = unmatched_points[counter];

          // remove points from processing that are already at the interface
          unmatched_points_idx.swap(unmatched_points_idx_next);

          // if every point is close enough to the interface, we are finished
          n_unmatched_points = dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

          if (n_unmatched_points == 0)
            break;
        }

      max_tangential_distance = dealii::Utilities::MPI::max(max_tangential_distance, mpi_comm);

      if (n_unmatched_points > 0)
        {
          pcout << "WARNING: The tolerance of the tangential correction of " << n_unmatched_points
                << " points is not yet attained. Max tangential distance value: "
                << max_tangential_distance << " max tolerance: " << tol_distance << std::endl;
          return false;
        }
      return true;
    }

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
    local_compute_normal_and_tangential_correction(std::vector<dealii::Point<dim>> &y)
    {
      AssertThrow(dim == 2 or dim == 3,
                  dealii::ExcMessage("Use local_compute_normal_correction for dim==1."));

      total_points_rpe = 0;

      // points that are not (yet) at the interface and still needs to be processed
      std::vector<unsigned int> unmatched_points_idx(y.size());
      std::iota(unmatched_points_idx.begin(), unmatched_points_idx.end(), 0);

      // temporary variable for signed distance
      std::vector<number>       evaluation_values_distance;
      std::vector<unsigned int> unmatched_points_normal_idx_next;

      for (int k = 0; k < additional_data.max_iter; ++k)
        {
          // correct entire points
          number max_tangential_distance = 0;
          int    n_unmatched_points =
            dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

          std::vector<unsigned int> unmatched_points_normal_and_tangential_idx_next;

          {
            std::ostringstream str;
            str << " k=" << k << " -> " << n_unmatched_points;
            if (additional_data.verbosity_level > 1)
              Journal::print_line(pcout, str.str(), "nearest_point");
          }

          // correct only unmatched points
          for (int j = 0; j < additional_data.max_iter; ++j)
            {
              std::vector<dealii::Point<dim>> unmatched_points(unmatched_points_idx.size());
              for (unsigned int i = 0; i < unmatched_points_idx.size(); ++i)
                unmatched_points[i] = y[unmatched_points_idx[i]];

              // just for output purposes
              std::ostringstream str;
              str << "   j=" << j << " "
                  << dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm) << " -> ";
              total_points_rpe += unmatched_points.size();

              remote_point_evaluation.reinit(unmatched_points,
                                             dof_handler_ls.get_triangulation(),
                                             mapping);

              AssertThrow(remote_point_evaluation.all_points_found(),
                          dealii::ExcMessage("Processed point is outside domain."));

              // compute signed distance at unmatched_points
              evaluation_values_distance =
                dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                     dof_handler_ls,
                                                     signed_distance);

              if (std::abs(additional_data.isocontour >= 0))
                {
                  for (auto &e : evaluation_values_distance)
                    e -= additional_data.isocontour;
                }

              // compute unit normal and tangent
              std::vector<dealii::Point<dim>>              evaluation_values_unit_normal;
              std::vector<std::vector<dealii::Point<dim>>> evaluation_values_unit_tangent;

              std::array<std::vector<number>, dim> evaluation_values_normal;

              for (int comp = 0; comp < dim; ++comp)
                {
                  evaluation_values_normal[comp] =
                    dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                         dof_handler_ls,
                                                         normal_vector.block(comp));
                }

              for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
                {
                  dealii::Point<dim> unit_normal;
                  for (unsigned int comp = 0; comp < dim; ++comp)
                    unit_normal[comp] = evaluation_values_normal[comp][counter];

                  const auto n_norm = unit_normal.norm();
                  unit_normal =
                    n_norm > tolerance_normal_vector ? unit_normal / n_norm : dealii::Point<dim>();

                  evaluation_values_unit_normal.emplace_back(unit_normal);
                  // compute the tangent(s) for each point
                  std::vector<dealii::Point<dim>> tangent;
                  tangent.resize(dim - 1);

                  if constexpr (dim == 2)
                    {
                      tangent[0][0] = unit_normal[1];
                      tangent[0][1] = -unit_normal[0];
                    }
                  else if constexpr (dim == 3)
                    {
                      dealii::Point<dim> temp_vec = dealii::Point<dim>::unit_vector(0);

                      // if normal vector is identical with unit vector
                      // choose different unit vector to compute the
                      // tangent
                      if ((temp_vec - unit_normal).norm() < 1e-10)
                        temp_vec = dealii::Point<dim>::unit_vector(1);

                      tangent[0] = temp_vec - (temp_vec * unit_normal) * unit_normal;
                      tangent[1] = cross_product_3d(unit_normal, tangent[0]);
                    }

                  evaluation_values_unit_tangent.emplace_back(tangent);
                }

              unmatched_points_normal_idx_next.clear();

              unsigned int n_complete = 0;

              for (unsigned int counter = 0; counter < unmatched_points.size(); ++counter)
                {
                  // perform tangential correction where normal correction was successful
                  if (std::abs(evaluation_values_distance[counter]) <= tol_distance)
                    {
                      // correct point by tangential offset
                      const auto distance_vec =
                        unmatched_points[counter] - stencil[unmatched_points_idx[counter]];

                      // determine tangential offset for each direction
                      std::vector<number> omega(dim - 1);
                      for (unsigned int d = 0; d < dim - 1; ++d)
                        omega[d] = distance_vec * evaluation_values_unit_tangent[counter][d];

                      // determine maximum tangential offset
                      number max_omega = std::numeric_limits<number>::lowest();
                      for (unsigned int d = 0; d < dim - 1; ++d)
                        max_omega = std::max(max_omega, std::abs(omega[d]));

                      max_tangential_distance = std::max(max_omega, max_tangential_distance);

                      if (max_omega > tol_distance)
                        {
                          unmatched_points_normal_and_tangential_idx_next.emplace_back(
                            unmatched_points_idx[counter]);

                          // correct point by tangential offset
                          for (unsigned int d = 0; d < dim - 1; ++d)
                            y[unmatched_points_idx[counter]] -=
                              omega[d] * evaluation_values_unit_tangent[counter][d];
                        }
                      else
                        n_complete += 1;
                    }
                  else
                    {
                      unmatched_points[counter] -= evaluation_values_distance[counter] *
                                                   evaluation_values_unit_normal[counter];

                      y[unmatched_points_idx[counter]] = unmatched_points[counter];

                      unmatched_points_normal_idx_next.emplace_back(unmatched_points_idx[counter]);
                    }
                }

              n_complete = dealii::Utilities::MPI::sum(n_complete, mpi_comm);


              // remove points from processing that are already at the interface
              unmatched_points_idx.swap(unmatched_points_normal_idx_next);

              // if every point is close enough to the interface, we are finished
              n_unmatched_points =
                dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm);

              str << n_unmatched_points << " (n ✗) "
                  << dealii::Utilities::MPI::sum(
                       unmatched_points_normal_and_tangential_idx_next.size(), mpi_comm)
                  << " (n ✓ | t ✗) " << n_complete << " (n ✓ | t ✓) ";
              if (additional_data.verbosity_level > 1)
                Journal::print_line(pcout, str.str(), "nearest_point", 10 /*special characters*/);

              if (n_unmatched_points == 0)
                break;
            }

          std::copy(unmatched_points_normal_idx_next.begin(),
                    unmatched_points_normal_idx_next.end(),
                    std::back_inserter(unmatched_points_normal_and_tangential_idx_next));
          std::sort(unmatched_points_normal_and_tangential_idx_next.begin(),
                    unmatched_points_normal_and_tangential_idx_next.end());

          unmatched_points_normal_and_tangential_idx_next.erase(
            std::unique(unmatched_points_normal_and_tangential_idx_next.begin(),
                        unmatched_points_normal_and_tangential_idx_next.end()),
            unmatched_points_normal_and_tangential_idx_next.end());

          unmatched_points_idx.swap(unmatched_points_normal_and_tangential_idx_next);

          // compute maximum distance of projected points to the level set 0 isosurface
          number max_distance = evaluation_values_distance.size() == 0 ?
                                  0.0 :
                                  *std::max_element(evaluation_values_distance.begin(),
                                                    evaluation_values_distance.end());

          max_distance = dealii::Utilities::MPI::max(max_distance, mpi_comm);

          if (n_unmatched_points > 0 and k == additional_data.max_iter - 1)
            {
              pcout << "WARNING: The tolerance of " << n_unmatched_points
                    << " points is not yet attained. Max distance value: " << max_distance
                    << std::endl;
              return false;
            }

          max_tangential_distance = dealii::Utilities::MPI::max(max_tangential_distance, mpi_comm);

          if (max_tangential_distance > tol_distance and k == additional_data.max_iter - 1)
            {
              pcout << "WARNING: The tolerance of the tangential correction of "
                    << dealii::Utilities::MPI::sum(unmatched_points_idx.size(), mpi_comm)
                    << " points is not yet attained. Max tangential distance value: "
                    << max_tangential_distance << " max tolerance: " << tol_distance << std::endl;
              return false;
            }
        }

      return true;
    }

    /**
     * Create a surface mesh and identify the closest point as the nearest vertex of the surface
     * mesh.
     */
    void
    local_compute_nearest_point()
    {
      std::unique_ptr<dealii::TimerOutput::Scope> timer_scope;
      if (timer_output)
        timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
          timer_output.value(), ScopedName("nearest_point::reinit::local::mca"));
      // create a point cloud of the surface
      dealii::GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(
        mapping,
        dof_handler_ls.get_fe(), // todo
        3 /*n subdivisions TODO: add parameter*/,
        1e-10 /*tolerance TODO: add parameter*/);

      std::vector<dealii::Point<dim>> surface_points;
      mc.process(dof_handler_ls, signed_distance, additional_data.isocontour, surface_points);

      // all gather surface points
      const auto surface_points_global =
        dealii::Utilities::MPI::all_gather(mpi_comm, surface_points);

      // TODO: find a faster way to get a single vector for all processes
      surface_points.clear();
      for (unsigned int i = 0; i < surface_points_global.size(); ++i)
        for (unsigned int j = 0; j < surface_points_global[i].size(); ++j)
          surface_points.emplace_back(surface_points_global[i][j]);

      if (timer_scope)
        timer_scope->stop();

      if (timer_output)
        timer_scope = std::make_unique<dealii::TimerOutput::Scope>(
          timer_output.value(), ScopedName("nearest_point::reinit::local::search"));

      const auto used_vertices_rtree = pack_rtree(surface_points);

      if (not used_vertices_rtree.empty())
        {
          // search for nearest point
          for (unsigned int i = 0; i < stencil.size(); ++i)
            {
              std::vector<dealii::Point<dim>> closest_vertex_in_domain;
              used_vertices_rtree.query(boost::geometry::index::nearest(stencil[i], 1),
                                        std::back_inserter(closest_vertex_in_domain));

              AssertThrow(closest_vertex_in_domain.size() == 1,
                          dealii::ExcMessage("The number of nearest points is wrong."));

              projected_points_at_interface[i] = closest_vertex_in_domain[0];
            }
        }
      if (timer_scope)
        timer_scope->stop();
    }

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


    // vectors to be filled: projected points to the interface corresponding to DoF indices
    std::vector<dealii::Point<dim>>                           projected_points_at_interface;
    std::vector<std::vector<dealii::types::global_dof_index>> dof_indices;
    std::vector<dealii::Point<dim>>                           stencil;

    bool is_reinit_called = false;

    // this is just a temporary variable to be called within the projection operators
    int total_points_rpe = 0;
  };
} // namespace MeltPoolDG::LevelSet::Tools
