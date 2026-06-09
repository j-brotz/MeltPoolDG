#pragma once
#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

namespace MeltPoolDG::LevelSet
{
  template <int dim, typename number>
  class ReinitializationOperationBase
  {
  public:
    virtual ~ReinitializationOperationBase() = default;

    virtual void
    solve() = 0;

    virtual void
    reinit() = 0;

    virtual void
    set_initial_condition(
      const dealii::LinearAlgebra::distributed::Vector<number> &solution_level_set_in) = 0;

    virtual void
    set_initial_condition(const dealii::Function<dim> & /*initial_field_function*/) = 0;

    /**
     * @brief Sets the boundary IDs where wetting boundary conditions should be applied.
     *
     * This function forwards the provided list of boundary IDs to the reinitialization
     * operation, indicating which boundaries are subject to wetting conditions.
     *
     * @param[in] wetting_bc_ids_in A list of boundary IDs where wetting conditions apply.
     *
     * @remark This function is not implemented for the base class.
     */
    virtual void
    set_wetting_boundary_condition_ids(
      std::vector<dealii::types::boundary_id> && /*wetting_bc_ids*/)
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    }

    virtual const dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() const = 0;

    virtual dealii::LinearAlgebra::distributed::Vector<number> &
    get_level_set() = 0;

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

  template <int dim, typename number>
  class ReinitializationHyperbolicOperationCapable
  {
  public:
    virtual ~ReinitializationHyperbolicOperationCapable() = default;

    virtual number
    get_max_change_level_set() const = 0;
  };

} // namespace MeltPoolDG::LevelSet
