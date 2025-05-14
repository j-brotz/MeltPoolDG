#pragma once
#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class ReinitializationOperationBase
  {
  public:
    virtual void
    solve() = 0;

    virtual void
    reinit() = 0;

    virtual void
    update_dof_idx(const unsigned int &reinit_dof_idx_in) = 0;

    virtual void
    set_initial_condition(
      const dealii::LinearAlgebra::distributed::Vector<number> &solution_level_set_in) = 0;

    virtual void
    set_initial_condition(const dealii::Function<dim> & /*initial_field_function*/) = 0;

    /**
     *
     * @param p_wetting_bc_map TODO AA
     */
    virtual void
    set_wetting_bc_map(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        & /*p_wetting_bc_map*/) = 0;
    /**
     *
     * @param p_wetting_bc_map TODO AA
     */
    virtual void
    set_contact_angle_bc_map(
      const std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        & /*p_contact_angle_bc_map*/) = 0;

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() = 0;

    virtual number
    get_max_change_level_set() const = 0;

    virtual const dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() const = 0;

    virtual dealii::LinearAlgebra::distributed::BlockVector<number> &
    get_normal_vector() = 0;

    virtual void
    attach_vectors(std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vectors) = 0;

    virtual void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const = 0;

    virtual number
    compute_CFL_based_timestep() const
    {
      AssertThrow(false,
                  dealii::ExcMessage(
                    "CFL based time stepping is not implemented for continous elements!"));
    };

    virtual void
    set_artificial_diffusitivity()
    {
      /**
       * Does nothing in the CG case
       */
    }

    virtual dealii::LinearAlgebra::distributed::Vector<number> *
    get_sign_indicator_function()
    {
      /**
       * Is not needed in the CG case
       */
      return nullptr;
    }
  };
} // namespace MeltPoolDG::LevelSet
