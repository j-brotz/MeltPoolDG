#pragma once

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/level_set/advection_diffusion_operation.hpp>
#include <meltpooldg/level_set/advection_diffusion_operation_base.hpp>
#include <meltpooldg/level_set/curvature_operation_base.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/level_set/level_set_operation_base.hpp>
#include <meltpooldg/level_set/nearest_point.hpp>
#include <meltpooldg/level_set/reinitialization_operation_base.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>


namespace MeltPoolDG::LevelSet
{
  /*
   *     Level set model including advection, reinitialization and curvature computation
   *     of the level set function.
   */
  template <int dim, typename number>
  class LevelSetOperation : public LevelSetOperationBase<dim, number>
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const ScratchData<dim, dim, number> &scratch_data;

    // Time stepping of the overall problem
    const TimeIntegration::TimeIterator<number> &time_stepping;
    /*
     *  The following objects are the operations, which are performed for solving the
     *  level set equation.
     */
    std::shared_ptr<AdvectionDiffusionOperationBase<dim, number>> advec_diff_operation;
    std::shared_ptr<ReinitializationOperationBase<dim, number>>   reinit_operation;
    std::shared_ptr<CurvatureOperationBase<dim, number>>          curvature_operation;
    /*
     *  necessary parameters
     */
    const LevelSetData<number> level_set_data;
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
    TimeIntegration::TimeIterator<number> reinit_time_iterator;

    bool ready_for_time_advance = false;
    /*
     *    This is the surface_tension vector calculated after level set and reinitialization
     * update
     */
    VectorType level_set_as_heaviside;
    VectorType distance_to_level_set;

    number max_d_level_set_since_last_reinit = std::numeric_limits<number>::max();


    // triangulation info on surface mesh of zero level set contour
    using SurfaceMeshInfo =
      std::vector<std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                             std::vector<dealii::Point<dim>> /*quad_points*/,
                             std::vector<number> /*weights*/
                             >>;
    SurfaceMeshInfo surface_mesh_info;

    std::unique_ptr<Tools::NearestPoint<dim, number>> nearest_point_search;

  public:
    LevelSetOperation(
      const ScratchData<dim, dim, number>                              &scratch_data_in,
      const TimeIntegration::TimeIterator<number>                      &time_stepping,
      const BoundaryConditionManager<dim, number>                      &bc_manager,
      [[maybe_unused]] const TimeIntegration::TimeSteppingData<number> &time_stepping_data,
      const LevelSetData<number>                                       &ls,
      const VectorType                                                 &advection_velocity,
      const unsigned int                                                ls_dof_idx_in,
      const unsigned int                                                ls_hanging_nodes_dof_idx_in,
      const unsigned int                                                ls_quad_idx_in,
      const unsigned int                                                reinit_dof_idx_in,
      const unsigned int                                                curv_dof_idx_in,
      const unsigned int                                                normal_dof_idx_in,
      const unsigned int                                                vel_dof_idx,
      const unsigned int                                                ls_zero_bc_idx);

    /**
     * set initial condition
     */
    void
    set_initial_condition(const dealii::Function<dim> &initial_field_function_level_set,
                          const bool is_signed_distance_initial_field_function = false) override;

    void
    set_inflow_outflow_bc(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        inflow_outflow_bc) override;

    void
    reinit() override;

    void
    distribute_constraints() override;

    void
    init_time_advance() override;

    void
    solve(const bool do_finish_time_step = true) override;

    void
    finish_time_advance() override;

    void
    set_level_set_user_rhs(const VectorType &level_set_user_rhs) override;

    void
    update_normal_vector() override;

    /*
     *  getter functions for solution vectors
     */
    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() override;

    const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() const override;

    dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set_as_heaviside() const override;

    dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set_as_heaviside() override;

    const dealii::LinearAlgebra::distributed::Vector<number> &
    get_distance_to_level_set() const override;

    const SurfaceMeshInfo &
    get_surface_mesh_info() const override;
    /**
     * register vectors for adaptive mesh refinement
     */
    void
    attach_vectors(
      std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const override;

    void
    update_surface_mesh() override;

    void
    transform_level_set_to_smooth_heaviside() override;

  private:
    void
    do_reinitialization(const bool update_normal_vector_in_every_cycle = false);

    inline number
    approximate_distance_from_level_set(const number phi, const number eps, const number cutoff)
    {
      if (std::abs(phi) < cutoff)
        return eps * std::log((1. + phi) / (1. - phi));
      else if (phi >= cutoff)
        return eps * std::log((1. + cutoff) / (1. - cutoff));
      else /*( phi <= -cutoff )*/
        return -eps * std::log((1. + cutoff) / (1. - cutoff));
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

    std::vector<dealii::Point<dim>> all_marked_vertices;
  };
} // namespace MeltPoolDG::LevelSet
