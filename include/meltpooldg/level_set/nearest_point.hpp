#pragma once
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/scoped_name.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <boost/geometry/index/rtree.hpp>

namespace MeltPoolDG::LevelSet::Tools
{
  using namespace dealii;

  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  /**
   * Compute nearest points to the isocontour of a level set function
   *
   * Based on a level set function and a cloud of points in the domain (stencil), compute
   * the corresponding nearest points to a discrete representation of the interface (isocontour).
   * The stencil is computed considering nodal points of a DoFHandler lying within a narrow band.
   * We support three types of algorithms (@p NearestPointType):
   *    - nearest_point: discretize the surface via the marching cube algorithm and take the closest
   *      point to the surface (cheap!)
   *    - closest_point_normal: iteratively correct the nearest point following the normal direction
   *      of the point.
   *    - closest_point_normal_collinear: extension of closest_point_normal to also ensure that the
   *      closest point is collinear to the interface.
   */
  template <int dim>
  class NearestPoint
  {
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
    NearestPoint(const Mapping<dim> &                               mapping,
                 const DoFHandler<dim> &                            dof_handler_signed_distance,
                 const VectorType &                                 signed_distance,
                 const BlockVectorType &                            normal_vector,
                 Utilities::MPI::RemotePointEvaluation<dim, dim> &  remote_point_evaluation,
                 const NearestPointData<double> &                   additional_data,
                 std::optional<std::reference_wrapper<TimerOutput>> timer_output = {})
      : mapping(mapping)
      , dof_handler_ls(dof_handler_signed_distance)
      , signed_distance(signed_distance)
      , normal_vector(normal_vector)
      , additional_data(additional_data)
      , remote_point_evaluation(remote_point_evaluation)
      , tol_distance(additional_data.rel_tol *
                     GridTools::minimal_cell_diameter(dof_handler_ls.get_triangulation()) /
                     std::sqrt(dim))
      , narrow_band_threshold(additional_data.narrow_band_threshold > 0 ?
                                additional_data.narrow_band_threshold :
                                signed_distance.linfty_norm() * 0.9999)
      , tolerance_normal_vector(
          UtilityFunctions::compute_numerical_zero_of_norm<dim>(dof_handler_ls.get_triangulation(),
                                                                mapping))
      , mpi_comm(dof_handler_ls.get_communicator())
      , timer_output(timer_output)
    {
      AssertThrow(dim <= 2 ||
                    additional_data.type != NearestPointType::closest_point_normal_collinear,
                  ExcNotImplemented());
    }

    /**
     * Update the nearest points of the nodal points from a given DoFHandler @p dof_handler_req.
     */
    void
    reinit(const DoFHandler<dim> &dof_handler_req)
    {
      ScopedName                          sc2("nearest_point::reinit");
      std::unique_ptr<TimerOutput::Scope> timer_scope;
      if (timer_output)
        timer_scope = std::make_unique<TimerOutput::Scope>(timer_output.value(), sc2);

      is_reinit_called = true;
      // calculate point cloud corresponding to nodes of the requested DoFHandler
      // located within the narrow band region
      const unsigned int n_components = dof_handler_req.get_fe().n_components();

      signed_distance.update_ghost_values();
      normal_vector.update_ghost_values();

      FEValues<dim> distance_values(
        mapping,
        dof_handler_ls.get_fe(),
        Quadrature<dim>(dof_handler_req.get_fe().base_element(0).get_unit_support_points()),
        update_values);

      FEValues<dim> req_values(
        mapping,
        dof_handler_req.get_fe(),
        Quadrature<dim>(dof_handler_req.get_fe().base_element(0).get_unit_support_points()),
        update_quadrature_points);

      // fill initial evaluation points with node coordinates (stencil)
      const unsigned int                   n_q_points = distance_values.get_quadrature().size();
      std::vector<double>                  temp_distance(n_q_points);
      std::vector<types::global_dof_index> temp_local_dof_indices(n_q_points * n_components);

      typename DoFHandler<dim>::active_cell_iterator req_cell = dof_handler_req.begin_active();

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
                  std::vector<types::global_dof_index> dofs_at_q;

                  // collect points of narrow band
                  if (std::abs(temp_distance[q]) < narrow_band_threshold)
                    {
                      stencil.emplace_back(req_values.quadrature_point(q));

                      // collect dof indices
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

      if (additional_data.type == NearestPointType::nearest_point)
        {
          local_compute_nearest_point();
        }
      else
        {
          // compute_closest points
          for (int j = 0; j < additional_data.max_iter; ++j)
            {
              if (local_compute_normal_correction(
                    projected_points_at_interface,
                    /* print warning */
                    j == additional_data.max_iter - 1 ||
                      additional_data.type != NearestPointType::closest_point_normal_collinear) ||
                  additional_data.type == NearestPointType::closest_point_normal)
                break;
            }
        }

      signed_distance.zero_out_ghost_values();
      normal_vector.zero_out_ghost_values();

      AssertThrow(dof_indices.size() == projected_points_at_interface.size(),
                  ExcMessage("The size of vectors does not match."));

      // TODO: where to update remote points (?)
      remote_point_evaluation.reinit(projected_points_at_interface,
                                     dof_handler_ls.get_triangulation(),
                                     mapping);
    }

    /**
     * Getter function for the nearest points, corresponding to the nodal points of the
     * DoFHandler passed into the reinit() function.
     *
     * @note Make sure that you have called reinit() before.
     */
    const std::vector<Point<dim>> &
    get_points() const
    {
      AssertThrow(is_reinit_called, ExcMessage("You need to call reinit() first."));

      return projected_points_at_interface;
    }

    /**
     * Getter function for the DoF indices of the nearest points,  corresponding to the nodal points
     * of the DoFHandler passed into the reinit() function.
     *
     * @note Make sure that you have called reinit() before.
     */
    const std::vector<std::vector<types::global_dof_index>> &
    get_dof_indices() const
    {
      AssertThrow(is_reinit_called, ExcMessage("You need to call reinit() first."));

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
      VectorType &                               solution_out,
      const DoFHandler<dim> &                    dof_handler_req, // TODO: remove
      const VectorType &                         solution_in,
      const bool                                 zero_out  = false,
      const std::function<double(const double)> &operation = {}) const
    {
      ScopedName                          sc("nearest_point::fill_dof_vector");
      std::unique_ptr<TimerOutput::Scope> timer_scope;
      if (timer_output)
        timer_scope = std::make_unique<TimerOutput::Scope>(timer_output.value(), sc);

      AssertThrow(n_components == dof_handler_req.get_fe().n_components(),
                  ExcMessage("There is a mismatch in the number of components "
                             "between your passed DoFHandler and the template parameter."));

      AssertThrow(is_reinit_called, ExcMessage("You need to call reinit() first."));

      const bool update_ghosts = !solution_in.has_ghost_elements();

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
                  solution_out[dof_indices[i][d]] = !operation ? vals[i][d] : operation(vals[i][d]);
            }
          else if (solution_out.locally_owned_elements().is_element(dof_indices[i][0]))
            solution_out[dof_indices[i][0]] = !operation ? vals[i] : operation(vals[i]);
        }
    }


    /**
     * Write the nearest points, calculated via reinit(), to a table file @p filename.
     */
    void
    write_to_file(const std::string filename = "closest_points") const
    {
      const auto global_points_normal_to_interface_all =
        Utilities::MPI::reduce<std::vector<Point<dim>>>(
          projected_points_at_interface, mpi_comm, [](const auto &a, const auto &b) {
            auto result = a;
            result.insert(result.end(), b.begin(), b.end());
            return result;
          });

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

  private:
    /**
     * Perform a closest point projection to the surface.
     */
    bool
    local_compute_normal_correction(std::vector<Point<dim>> &y, const bool print_warning)
    {
      // points that are not (yet) at the interface and still needs to be processed
      std::vector<Point<dim>>   closest_points = y;
      std::vector<unsigned int> closest_points_idx(closest_points.size());

      std::iota(closest_points_idx.begin(), closest_points_idx.end(), 0);

      // temporary variable for signed distance
      std::vector<double> evaluation_values_distance;

      double max_tangential_distance = 0;
      int    n_closest_points        = Utilities::MPI::sum(closest_points.size(), mpi_comm);
      int    n_omega                 = 0;

      for (int j = 0; j < additional_data.max_iter; ++j)
        {
          remote_point_evaluation.reinit(closest_points,
                                         dof_handler_ls.get_triangulation(),
                                         mapping);

          AssertThrow(remote_point_evaluation.all_points_found(),
                      ExcMessage("Processed point is outside domain."));

          // compute signed distance at closest_points
          evaluation_values_distance = dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                                            dof_handler_ls,
                                                                            signed_distance);

          if (std::abs(additional_data.isocontour >= 0))
            {
              for (auto &e : evaluation_values_distance)
                e -= additional_data.isocontour;
            }

          // compute unit normal and tangent
          std::vector<Point<dim>> evaluation_values_unit_normal;
          std::vector<Point<dim>> evaluation_values_unit_tangent;

          std::array<std::vector<double>, dim> evaluation_values_normal;

          for (int comp = 0; comp < dim; ++comp)
            {
              evaluation_values_normal[comp] =
                dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                     dof_handler_ls,
                                                     normal_vector.block(comp));
            }

          for (unsigned int counter = 0; counter < closest_points.size(); ++counter)
            {
              Point<dim> unit_normal;
              for (unsigned int comp = 0; comp < dim; ++comp)
                unit_normal[comp] = evaluation_values_normal[comp][counter];

              const auto n_norm = unit_normal.norm();
              unit_normal =
                (n_norm > tolerance_normal_vector) ? unit_normal / n_norm : Point<dim>();

              evaluation_values_unit_normal.emplace_back(unit_normal);

              if (dim > 1 &&
                  additional_data.type == NearestPointType::closest_point_normal_collinear)
                {
                  Point<dim> tangent;
                  // TODO: add 3D tangent
                  if constexpr (dim == 2)
                    {
                      tangent[0] = unit_normal[1];
                      tangent[1] = -unit_normal[0];
                      evaluation_values_unit_tangent.emplace_back(tangent);
                    }
                }
            }

          std::vector<Point<dim>>   closest_points_new;
          std::vector<unsigned int> closest_points_idx_new;

          // counter for points that need to be corrected tangentially
          n_omega = 0;

          for (unsigned int counter = 0; counter < closest_points.size(); ++counter)
            {
              // skip point where the distance to the interface is already close enough
              if (std::abs(evaluation_values_distance[counter]) <= tol_distance)
                {
                  if (additional_data.type == NearestPointType::closest_point_normal_collinear)
                    {
                      // correct point by tangential offset
                      const auto distance_vec =
                        closest_points[counter] - stencil[closest_points_idx[counter]];

                      const double omega = distance_vec * evaluation_values_unit_tangent[counter];

                      if (omega > tol_distance)
                        n_omega++;

                      max_tangential_distance = std::max(std::abs(omega), max_tangential_distance);

                      if constexpr (dim == 2)
                        y[closest_points_idx[counter]] -=
                          omega * evaluation_values_unit_tangent[counter];
                    }
                  continue;
                }

              // compute closest point and update value in global vector
              closest_points[counter] -=
                evaluation_values_distance[counter] * evaluation_values_unit_normal[counter];

              y[closest_points_idx[counter]] = closest_points[counter];

              closest_points_new.emplace_back(closest_points[counter]);
              closest_points_idx_new.emplace_back(closest_points_idx[counter]);
            }

          // remove points from processing that are already at the interface
          closest_points.swap(closest_points_new);
          closest_points_idx.swap(closest_points_idx_new);

          // if every point is close enough to the interface, we are finished
          n_closest_points = Utilities::MPI::sum(closest_points.size(), mpi_comm);

          if (n_closest_points == 0)
            break;
        }

      // compute maximum distance of projected points to the level set 0 isosurface
      double max_distance =
        (evaluation_values_distance.size() == 0) ?
          0.0 :
          *std::max_element(evaluation_values_distance.begin(), evaluation_values_distance.end());

      max_distance = Utilities::MPI::max(max_distance, mpi_comm);

      dealii::ConditionalOStream pcout(std::cout,
                                       Utilities::MPI::this_mpi_process(mpi_comm) == 0 &&
                                         print_warning);

      if (n_closest_points > 0 && max_distance > tol_distance)
        {
          pcout << "WARNING: The tolerance of " << n_closest_points
                << " points is not yet attained. Max distance value: " << max_distance << std::endl;
          return false;
        }

      if (additional_data.type == NearestPointType::closest_point_normal_collinear)
        {
          max_tangential_distance = Utilities::MPI::max(max_tangential_distance, mpi_comm);
          n_omega                 = Utilities::MPI::sum(n_omega, mpi_comm);

          if (max_tangential_distance > tol_distance)
            {
              pcout << "WARNING: The tolerance of the tangential correction of " << n_omega
                    << " points is not yet attained. Max tangential distance value: "
                    << max_tangential_distance << " max tolerance: " << tol_distance << std::endl;
              return false;
            }
        }

      return true;
    }

    /**
     * Create a surface mesh and find the nearest vertices to the surface mesh.
     */
    void
    local_compute_nearest_point()
    {
      // create a point cloud of the surface
      GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(mapping,
                                                           dof_handler_ls.get_fe(), // todo
                                                           3 /*n subdivisions TODO: add parameter*/,
                                                           1e-10 /*tolerance TODO: add parameter*/);

      std::vector<Point<dim>> surface_points;
      mc.process(dof_handler_ls, signed_distance, additional_data.isocontour, surface_points);

      const auto used_vertices_rtree = pack_rtree(surface_points);

      if (!used_vertices_rtree.empty())
        {
          // search for nearest point
          for (unsigned int i = 0; i < stencil.size(); ++i)
            {
              std::vector<Point<dim>> closest_vertex_in_domain;
              used_vertices_rtree.query(boost::geometry::index::nearest(stencil[i], 1),
                                        std::back_inserter(closest_vertex_in_domain));

              AssertThrow(closest_vertex_in_domain.size() == 1,
                          ExcMessage("The number of nearest points is wrong."));

              projected_points_at_interface[i] = closest_vertex_in_domain[0];
            }
        }
    }

    const Mapping<dim> &            mapping;
    const DoFHandler<dim> &         dof_handler_ls;
    const VectorType &              signed_distance;
    const BlockVectorType &         normal_vector;
    const NearestPointData<double> &additional_data;

    Utilities::MPI::RemotePointEvaluation<dim, dim> &remote_point_evaluation;

    // Tolerance to be reached for the distance of the projected points to the distance = 0
    // isosurface
    const double tol_distance;
    // In the default case, we limit the interval for closest point projection to
    // max(distance)*0.9999 to avoid projection in regions, where the distance is constant at
    // max(distance).
    const double narrow_band_threshold;

    const double tolerance_normal_vector;

    const MPI_Comm mpi_comm;

    std::optional<std::reference_wrapper<TimerOutput>> timer_output;

    // vectors to be filled: projected points to the interface corresponding to DoF indices
    std::vector<Point<dim>>                           projected_points_at_interface;
    std::vector<std::vector<types::global_dof_index>> dof_indices;
    std::vector<Point<dim>>                           stencil;

    bool is_reinit_called = false;
  };
} // namespace MeltPoolDG::LevelSet::Tools
