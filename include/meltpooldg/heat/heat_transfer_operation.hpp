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
#include <meltpooldg/heat/heat_transfer_preconditioner_matrixfree.hpp>
#include <meltpooldg/interface/periodic_boundary_conditions.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/utilities/generic_data_out.hpp>
#include <meltpooldg/utilities/newton_raphson_solver.hpp>
#include <meltpooldg/utilities/time_iterator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;

  template <int dim>
  class HeatTransferOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim> &                 scratch_data;
    std::shared_ptr<BoundaryConditions<dim>> bc_data;
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
    VectorType user_rhs;
    VectorType temperature_interface;

    // optional flow velocity for internal convection
    const unsigned int vel_dof_idx;
    VectorType *       velocity;

    // optional level-set heaviside field for two phase flow
    const unsigned int ls_dof_idx;
    VectorType *       level_set_as_heaviside;

    std::shared_ptr<HeatTransferOperator<dim>> heat_operator;

    std::shared_ptr<HeatTransferPreconditionerMatrixFree<dim>> heat_transfer_preconditioner;
    std::shared_ptr<DiagonalMatrix<VectorType>>                diag_preconditioner;
    std::shared_ptr<TrilinosWrappers::PreconditionBase>        trilinos_preconditioner;

  public:
    HeatTransferOperation(std::shared_ptr<BoundaryConditions<dim>> bc_data,
                          const ScratchData<dim> &                 scratch_data_in,
                          const HeatData<double> &                 heat_data_in,
                          const Material<double> &                 material,
                          unsigned int                             temp_dof_idx_in,
                          unsigned int                             temp_hanging_nodes_dof_idx_in,
                          unsigned int                             temp_quad_idx_in,
                          unsigned int                             vel_dof_idx_in = 0,
                          VectorType *                             velocity_in    = nullptr,
                          unsigned int                             ls_dof_idx_in  = 0,
                          VectorType *level_set_as_heaviside_in                   = nullptr);

    void
    register_evaporative_mass_flux(VectorType *       evaporative_mass_flux_in,
                                   const unsigned int evapor_mass_flux_dof_idx_in,
                                   const double       latent_heat_of_evaporation,
                                   const bool         do_phenomenological_recoil_pressure);

    void
    set_initial_condition(const Function<dim> &initial_field_function_temperature,
                          const double         start_time);

    void
    reinit();

    void
    solve(const TimeIterator<double> &time_iterator);

    void
    compute_interface_temperature(const VectorType &distance, const BlockVectorType &normal_vector);

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
    get_temperature_interface() const;

    VectorType &
    get_temperature_interface();

    const VectorType &
    get_heat_source() const;

    VectorType &
    get_heat_source();

    const VectorType &
    get_user_rhs() const;

    VectorType &
    get_user_rhs();

    const VectorType &
    get_level_set_as_heaviside() const;
  };
} // namespace MeltPoolDG::Heat
