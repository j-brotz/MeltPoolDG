#pragma once
#include <deal.II/lac/la_parallel_block_vector.h>

#include <meltpooldg/core/periodic_boundary_conditions.hpp>
#include <meltpooldg/level_set/advection_diffusion_data.hpp>
#include <meltpooldg/post_processing/generic_data_out.hpp>


namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class AdvectionDiffusionOperationBase
  {
  protected:
    /// determine whether solution vectors are prepared for time advance
    bool ready_for_time_advance = false;

  public:
    virtual ~AdvectionDiffusionOperationBase() = default;

    AdvectionDiffusionData<number> advec_diff_data;

    virtual void
    init_time_advance() = 0;

    virtual void
    finish_time_advance()
    {
      ready_for_time_advance = false;
    }

    virtual void
    solve(const bool do_finish_time_step = true) = 0;

    virtual void
    reinit() = 0;

    virtual void
    set_initial_condition(const dealii::Function<dim> & /*initial_field_function*/) = 0;

    virtual void
    set_advection_velocity(const dealii::LinearAlgebra::distributed::Vector<number> &,
                           const unsigned int /*dof_idx*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    virtual void
    set_advection_velocity_function(
      const std::shared_ptr<dealii::Function<dim, number>> & /*advection_velocity*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    virtual void
    setup_constraints(ScratchData<dim, dim, number>                          &mutable_scratch_data,
                      const PeriodicBoundaryConditions<dim>                  &pbc,
                      const std::map<dealii::types::boundary_id,
                                     std::shared_ptr<dealii::Function<dim>>> &dirichlet_bc_in) = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_advected_field_old() = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_user_rhs() const = 0;

    virtual void
    attach_vectors(std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    virtual void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;
  };
} // namespace MeltPoolDG::LevelSet
