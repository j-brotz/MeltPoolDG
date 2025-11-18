#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <map>
#include <memory>
#include <tuple>
#include <vector>


namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class LevelSetOperationBase
  {
    // triangulation info on surface mesh of zero level set contour
    using SurfaceMeshInfo =
      std::vector<std::tuple<const typename dealii::Triangulation<dim, dim>::cell_iterator /*cell*/,
                             std::vector<dealii::Point<dim>> /*quad_points*/,
                             std::vector<number> /*weights*/
                             >>;

  public:
    virtual void
    set_initial_condition(const dealii::Function<dim> &initial_field_function_level_set,
                          const bool is_signed_distance_initial_field_function = false) = 0;

    virtual void
    set_inflow_outflow_bc(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        inflow_outflow_bc) = 0;

    virtual void
    setup_constraints(
      ScratchData<dim, dim, number>         &mutable_scratch_data,
      const PeriodicBoundaryConditions<dim> &pbc,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &ls_dirichlet_bc_in,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &normal_x_dirichlet_bc_in,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &normal_y_dirichlet_bc_in,
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        &normal_z_dirichlet_bc_in) = 0;

    virtual void
    reinit() = 0;

    virtual void
    distribute_constraints() = 0;


    virtual void
    init_time_advance() = 0;

    virtual void
    solve(const bool do_finish_time_step = true) = 0;

    virtual void
    finish_time_advance() = 0;

    virtual void
    set_level_set_user_rhs(
      const dealii::LinearAlgebra::distributed::Vector<number> &level_set_user_rhs) = 0;

    virtual void
    set_wetting_boundary_condition_ids(
      std::vector<dealii::types::boundary_id> && /*wetting_bc_ids*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    virtual void
    update_normal_vector() = 0;

    /*
     *  getter functions for solution vectors
     */
    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_curvature() = 0;

    virtual const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() const = 0;

    virtual dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set_as_heaviside() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set_as_heaviside() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_distance_to_level_set() const = 0;

    virtual const SurfaceMeshInfo &
    get_surface_mesh_info() const = 0;
    /**
     * register vectors for adaptive mesh refinement
     */
    virtual void
    attach_vectors(std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    virtual void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;

    virtual void
    update_surface_mesh() = 0;

    virtual void
    transform_level_set_to_smooth_heaviside() = 0;
  };
} // namespace MeltPoolDG::LevelSet
