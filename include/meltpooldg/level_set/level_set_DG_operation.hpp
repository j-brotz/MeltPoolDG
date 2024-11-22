/* ---------------------------------------------------------------------
 * Johannes Resch, TUM, July 2024
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/advection_diffusion/advection_DG_operation.hpp>
#include <meltpooldg/curvature/curvature_DG_operation.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/level_set/level_set_data.hpp>
#include <meltpooldg/level_set/level_set_operation_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_DG_operation.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>
#include <meltpooldg/reinitialization/reinitialization_DG_operation.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>

#include <limits>
#include <map>
#include <memory>
#include <vector>


namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  /*
   *     Level set model including advection, reinitialization and curvature computation
   *     of the level set function.
   */
  template <int dim>
  class LevelSetDGOperation : public LevelSetOperationBase<dim>
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
    std::shared_ptr<AdvectionDGOperation<dim>>        advec_operation;
    std::shared_ptr<ReinitializationDGOperation<dim>> reinit_operation;

    // Is used to track the unreinitialized interface movement
    std::shared_ptr<AdvectionDGOperation<dim>> advec_smoothed_signum_operation;

    /*
     *   Computation of the normal vectors
     */
    std::shared_ptr<NormalVectorOperationBase<dim>> normal_vector_operation;
    /*
     *   Computation of the curvature
     */
    std::shared_ptr<CurvatureDGOperation<dim>> curvature_operation;
    /*
     *  necessary parameters
     */
    const LevelSetData<double> &level_set_data;
    /*
     * select the relevant DoFHandler
     */
    const unsigned int ls_dof_idx;
    const unsigned int ls_quad_idx;
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

    double max_d_level_set_since_last_reinit = std::numeric_limits<double>::max();


    // triangulation info on surface mesh of zero level set contour
    using SurfaceMeshInfo =
      std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                             std::vector<Point<dim>> /*quad_points*/,
                             std::vector<double> /*weights*/
                             >>;
    SurfaceMeshInfo surface_mesh_info;

    int iter = 0;

  public:
    LevelSetDGOperation(const ScratchData<dim>                        &scratch_data_in,
                        const TimeIterator<double>                    &time_stepping,
                        std::shared_ptr<SimulationParametersBase<dim>> base_in,
                        VectorType                                    &advection_velocity,
                        const unsigned int                             ls_dof_idx_in,
                        const unsigned int                             ls_quad_idx_in,
                        const unsigned int                             reinit_dof_idx_in,
                        const unsigned int                             vel_dof_idx);
    /**
     * set initial condition
     */
    void
    set_initial_condition(const Function<dim> &initial_field_function_level_set,
                          const bool is_signed_distance_initial_field_function = false) override;

    void
    set_inflow_outflow_bc(
      [[maybe_unused]] const std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
        inflow_outflow_bc) override
    { // Not needed in the DG case sinde BCs are applied weakly within the operator
      DEAL_II_NOT_IMPLEMENTED();
    }

    void
    reinit() override;

    void
    distribute_constraints() override
    {
      // Not needed in the DG case since constraints are applied in a weak sense
      DEAL_II_NOT_IMPLEMENTED();
    };

    void
    init_time_advance() override;

    void
    solve(const bool do_finish_time_step = true) override;

    void
    finish_time_advance() override;

    void
    set_level_set_user_rhs([[maybe_unused]] const VectorType &level_set_user_rhs) override
    {
      AssertThrow(
        false,
        ExcMessage(
          "The function set_level_set_user_rhs function is not implemented for DG level set."));
    }

    void
    update_normal_vector() override
    {
      AssertThrow(
        false,
        ExcMessage(
          "The function update_normal_vector function is not implemented for DG level set."));
    }

    /*
     *  getter functions for solution vectors
     */
    const LinearAlgebra::distributed::Vector<double> &
    get_curvature() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_curvature() override;

    const LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() const override;

    LinearAlgebra::distributed::BlockVector<double> &
    get_normal_vector() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_level_set() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_level_set() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_level_set_as_heaviside() const override;

    LinearAlgebra::distributed::Vector<double> &
    get_level_set_as_heaviside() override;

    const LinearAlgebra::distributed::Vector<double> &
    get_distance_to_level_set() const override;

    const SurfaceMeshInfo &
    get_surface_mesh_info() const override;

    /**
     * register vectors for adaptive mesh refinement
     */
    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override;

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const override;

    void
    update_surface_mesh() override;

    void
    transform_level_set_to_smooth_heaviside() override;

  private:
    void
    do_reinitialization(const bool update_normal_vector_in_every_cycle = false);

    // Computes a given error norm of the level set field
    double
    compute_level_set_gradient_error(const VectorType &solution);
  };
} // namespace MeltPoolDG::LevelSet
