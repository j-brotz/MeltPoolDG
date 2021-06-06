/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/data_out.h>

#include <meltpooldg/heat/heat_transfer_operator.hpp>
#include <meltpooldg/heat/heat_transfer_preconditioner.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
#include <meltpooldg/utilities/linear_solve.hpp>
#include <meltpooldg/utilities/newton_raphson_solver.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class HeatTransferOperation
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    const ScratchData<dim> &scratch_data;
    /**
     * parameters
     */
    const HeatData<double> heat_data;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int temp_dof_idx;
    const unsigned int temp_hanging_nodes_dof_idx;
    const unsigned int temp_quad_idx;
    /*
     *    This are the primary solution variables of this module, which will be also publically
     *    accessible for output_results.
     */
    VectorType temperature;
    VectorType temperature_old;
    VectorType heat_source;

    // optional flow velocity for internal convection
    const unsigned int vel_dof_idx;
    VectorType *       velocity;

    // optional level-set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    VectorType *       level_set_as_heaviside;

    std::shared_ptr<HeatTransferOperator<dim>> heat_operator;

    const MaterialData<double> &material_data;

    HeatTransferPreconditioner<dim> heat_transfer_preconditioner;

  public:
    HeatTransferOperation(const std::shared_ptr<BoundaryConditions<dim>> &bc_data,
                          const ScratchData<dim> &                        scratch_data_in,
                          const HeatData<double> &                        heat_data_in,
                          const MaterialData<double> &                    material_data,
                          unsigned int                                    temp_dof_idx_in,
                          unsigned int                   temp_hanging_nodes_dof_idx_in,
                          unsigned int                   temp_quad_idx_in,
                          unsigned int                   vel_dof_idx_in            = 0,
                          VectorType *                   velocity_in               = nullptr,
                          unsigned int                   ls_dof_idx_in             = 0,
                          VectorType *                   level_set_as_heaviside_in = nullptr,
                          const EvaporationData<double> *evapor_data_in            = nullptr);

    void
    register_evaporative_mass_flux(VectorType *evaporative_mass_flux_in);

    void
    set_initial_condition(const Function<dim> &initial_field_function_temperature);

    void
    reinit();

    void
    solve(double dt);

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors);

    void
    distribute_constraints();

    void
    attach_output_vectors(GenericDataOut<dim> &data_out) const;

    const VectorType &
    get_temperature() const;

    VectorType &
    get_temperature();

    const VectorType &
    get_heat_source() const;

    VectorType &
    get_heat_source();
  };
} // namespace MeltPoolDG::Heat
