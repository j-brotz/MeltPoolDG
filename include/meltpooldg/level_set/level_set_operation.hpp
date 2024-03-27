/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operation_base.hpp>
#include <meltpooldg/curvature/curvature_operation_base.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  /*
   *     Level set model including advection, reinitialization and curvature computation
   *     of the level set function.
   */
  template <int dim>
  class LevelSetOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim> &scratch_data;

    // Time stepping of the overall problem
    const TimeIterator<double> &time_stepping;
    /*
     *  The following objects are the operations, which are performed for solving the
     *  level set equation.
     */
    std::shared_ptr<AdvectionDiffusionOperationBase<dim>> advec_diff_operation;
    std::shared_ptr<ReinitializationOperationBase<dim>>   reinit_operation;
    std::shared_ptr<CurvatureOperationBase<dim>>          curvature_operation;
    /*
     *  necessary parameters
     */
    const LevelSetData<double> level_set_data;
    /*
     * select the relevant DoFHandler
     */
    const unsigned int ls_dof_idx;
    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int ls_quad_idx;
    const unsigned int curv_dof_idx;
    const unsigned int reinit_dof_idx;
    /*
     *  The reinitialization of the level set function is a "pseudo"-time-dependent
     *  equation, which is solved up to quasi-steady state. Thus a time iterator is
     *  needed.
     */
    TimeIterator<double> reinit_time_iterator;

    bool ready_for_time_advance = false;
    /*
     *    This is the surface_tension vector calculated after level set and reinitialization
     * update
     */
    VectorType level_set_as_heaviside;
    VectorType distance_to_level_set;

    double max_d_level_set_since_last_reinit = std::numeric_limits<double>::max();


    // triangulation info on surface mesh of zero level set contour
    using SurfaceMeshInfo =
      std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                             std::vector<Point<dim>> /*quad_points*/,
                             std::vector<double> /*weights*/
                             >>;
    SurfaceMeshInfo surface_mesh_info;

    std::unique_ptr<Tools::NearestPoint<dim>> nearest_point_search;

  public:
    LevelSetOperation(const ScratchData<dim>              &scratch_data_in,
                      const TimeIterator<double>          &time_stepping,
                      std::shared_ptr<SimulationBase<dim>> base_in,
                      const VectorType                    &advection_velocity,
                      const unsigned int                   ls_dof_idx_in,
                      const unsigned int                   ls_hanging_nodes_dof_idx_in,
                      const unsigned int                   ls_quad_idx_in,
                      const unsigned int                   reinit_dof_idx_in,
                      const unsigned int                   curv_dof_idx_in,
                      const unsigned int                   normal_dof_idx_in,
                      const unsigned int                   vel_dof_idx,
                      const unsigned int                   ls_zero_bc_idx);

    /**
     * set initial condition
     */
    void
    set_initial_condition(const Function<dim> &initial_field_function_level_set,
                          const bool           is_signed_distance_initial_field_function = false);

    void
    set_inflow_outflow_bc(
      const std::map<types::boundary_id, std::shared_ptr<Function<dim>>> inflow_outflow_bc);

    void
    reinit();

    void
    distribute_constraints();

    void
    init_time_advance();

    void
    solve(const bool do_finish_time_step = true);

    void
    finish_time_advance();

    void
    set_level_set_user_rhs(const VectorType &level_set_user_rhs);

    void
    update_normal_vector();

    /*
     *  getter functions for solution vectors
     */
    const LinearAlgebra::distributed::Vector<double> &
    get_curvature() const;

    LinearAlgebra::distributed::Vector<double> &
    get_curvature();

    const LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() const;

    LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector();

    const LinearAlgebra::distributed::Vector<double> &
    get_level_set() const;

    LinearAlgebra::distributed::Vector<double> &
    get_level_set();

    const LinearAlgebra::distributed::Vector<double> &
    get_level_set_as_heaviside() const;

    LinearAlgebra::distributed::Vector<double> &
    get_level_set_as_heaviside();

    const LinearAlgebra::distributed::Vector<double> &
    get_distance_to_level_set() const;

    const SurfaceMeshInfo &
    get_surface_mesh_info() const;
    /**
     * register vectors for adaptive mesh refinement
     */
    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    void
    update_surface_mesh();

    void
    transform_level_set_to_smooth_heaviside();

  private:
    void
    do_reinitialization(const bool update_normal_vector_in_every_cycle = false);

    inline double
    approximate_distance_from_level_set(const double phi, const double eps, const double cutoff)
    {
      if (std::abs(phi) < cutoff)
        return eps * std::log((1. + phi) / (1. - phi));
      else if (phi >= cutoff)
        return eps * std::log((1. + cutoff) / (1. - cutoff));
      else /*( phi <= -cutoff )*/
        return -eps * std::log((1. + cutoff) / (1. - cutoff));
    }

    /**
     * The given distance value is transformed to a smooth heaviside function \f$H_\epsilon\f$,
     * which has the property of \f$\int \nabla H_\epsilon=1\f$. This function has its transition
     * region between -2 and 2.
     */
    inline double
    smooth_heaviside_from_distance_value(const double x /*distance*/)
    {
      if (x > 0)
        return 1. - smooth_heaviside_from_distance_value(-x);
      else if (x < -2.)
        return 0;
      else if (x < -1.)
        {
          const double x2 = x * x;
          return (
            0.125 * (5. * x + x2) + 0.03125 * (-3. - 2. * x) * std::sqrt(-7. - 12. * x - 4. * x2) -
            0.0625 * std::asin(std::sqrt(2.) * (x + 1.5)) + 23. * 0.03125 - numbers::PI / 64.);
        }
      else
        {
          const double x2 = x * x;
          return (
            0.125 * (3. * x + x2) - 0.03125 * (-1. - 2. * x) * std::sqrt(1. - 4. * x - 4. * x2) +
            0.0625 * std::asin(std::sqrt(2.) * (x + 0.5)) + 15. * 0.03125 - numbers::PI / 64.);
        }
    }

    /**
     * From the distance_to_level_set DoF vector, the level set DoF vector with values within
     * [-1,1] is computed.
     */
    void
    transform_distance_to_level_set();

    /// To avoid high-frequency errors in the curvature (spurious currents) the curvature is
    /// corrected to represent the value of the interface (zero level set). The approach by Zahedi
    /// et al. (2012) is pursued. Considering e.g. a bubble, the absolute curvature of areas
    /// outside of the bubble (Φ=-) must increase and vice-versa for areas
    ///  inside the bubble.
    //
    //           ******
    //       ****      ****
    //     **              **
    //    *      Φ=+         *  Φ=-
    //    *    sgn(d)=+      *  sgn(d)=-
    //    *                  *
    //     **              **
    //       ****      ****
    //           ******
    //
    void
    correct_curvature_values();

    void
    set_level_set_parameters();

    std::vector<Point<dim>> all_marked_vertices;
  };
} // namespace MeltPoolDG::LevelSet
