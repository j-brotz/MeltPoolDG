/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat/heat_transfer_operator.hpp>
#include <meltpooldg/heat/heat_transfer_preconditioner.hpp>
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
                          const unsigned int                              temp_dof_idx_in,
                          const unsigned int temp_hanging_nodes_dof_idx_in,
                          const unsigned int temp_quad_idx_in,
                          const unsigned int vel_dof_idx_in            = 0,
                          VectorType *       velocity_in               = nullptr,
                          const unsigned int ls_dof_idx_in             = 0,
                          VectorType *       level_set_as_heaviside_in = nullptr)
      : scratch_data(scratch_data_in)
      , heat_data(heat_data_in)
      , temp_dof_idx(temp_dof_idx_in)
      , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
      , temp_quad_idx(temp_quad_idx_in)
      , vel_dof_idx(vel_dof_idx_in)
      , velocity(velocity_in)
      , ls_dof_idx(ls_dof_idx_in)
      , level_set_as_heaviside(level_set_as_heaviside_in)
      , material_data(material_data)
      , heat_transfer_preconditioner(scratch_data, temp_dof_idx)
    {
      heat_operator = std::make_shared<HeatTransferOperator<dim>>(bc_data,
                                                                  scratch_data,
                                                                  heat_data,
                                                                  material_data,
                                                                  temp_dof_idx,
                                                                  temp_quad_idx,
                                                                  temperature,
                                                                  heat_source,
                                                                  vel_dof_idx,
                                                                  velocity,
                                                                  ls_dof_idx,
                                                                  level_set_as_heaviside);
    }

    void
    set_initial_condition(const Function<dim> &initial_field_function_temperature)
    {
      reinit();

      heat_transfer_preconditioner.reinit();

      dealii::VectorTools::project(scratch_data.get_mapping(),
                                   scratch_data.get_dof_handler(temp_dof_idx),
                                   scratch_data.get_constraint(temp_dof_idx),
                                   scratch_data.get_quadrature(temp_quad_idx),
                                   initial_field_function_temperature,
                                   temperature);
      temperature_old.copy_locally_owned_data_from(temperature);
    }

    void
    reinit()
    {
      scratch_data.initialize_dof_vector(temperature, temp_dof_idx);
      scratch_data.initialize_dof_vector(temperature_old, temp_dof_idx);
      scratch_data.initialize_dof_vector(heat_source, temp_dof_idx);
    }

    void
    solve(const double dt)
    {
      if (!heat_data.do_matrix_free)
        AssertThrow(false, ExcNotImplemented());

      heat_operator->set_time_increment(dt);
      temperature_old = temperature;

      const auto create_rhs = [&](VectorType &rhs) {
        heat_operator->create_rhs_and_apply_dirichlet_mf(
          rhs, temperature_old, scratch_data, temp_dof_idx, temp_hanging_nodes_dof_idx);
      };

      const auto solve_linear_system = [&](VectorType &      solution_update,
                                           const VectorType &rhs) -> int {
        if (heat_data.solver.preconditioner_type == "Identity")
          {
            return LinearSolve::solve<VectorType, SolverGMRES<VectorType>, OperatorBase<double>>(
              *heat_operator,
              solution_update,
              rhs,
              heat_data.solver.rel_tolerance,
              heat_data.solver.max_iterations);
          }
        else if (heat_data.solver.preconditioner_type == "Diagonal" ||
                 heat_data.solver.preconditioner_type == "DiagonalReduced")

          {
            auto preconditioner = heat_transfer_preconditioner.get_diagonal_preconditioner(
              heat_data.solver.preconditioner_type, heat_operator);

            return LinearSolve::solve<VectorType, SolverGMRES<VectorType>, OperatorBase<double>>(
              *heat_operator,
              solution_update,
              rhs,
              heat_data.solver.rel_tolerance,
              heat_data.solver.max_iterations,
              preconditioner);
          }
        else if (heat_data.solver.preconditioner_type == "AMG" ||
                 heat_data.solver.preconditioner_type == "AMGReduced")
          {
            heat_operator->compute_system_matrix(heat_transfer_preconditioner.get_system_matrix(),
                                                 heat_data.solver.preconditioner_type == "AMG");

            auto preconditioner =
              LinearSolve::setup_preconditioner(heat_transfer_preconditioner.get_system_matrix(),
                                                heat_data.solver.preconditioner_type);

            return LinearSolve::solve<VectorType, SolverGMRES<VectorType>, OperatorBase<double>>(
              *heat_operator,
              solution_update,
              rhs,
              heat_data.solver.rel_tolerance,
              heat_data.solver.max_iterations,
              *preconditioner);
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
            return 0;
          }
      };

      auto newton = NewtonRaphsonSolver<dim>(scratch_data,
                                             heat_data.nlsolve,
                                             temp_dof_idx,
                                             temp_quad_idx,
                                             temperature_old,
                                             temperature,
                                             create_rhs,
                                             solve_linear_system);

      newton.solve();
    }

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      temperature.update_ghost_values();
      vectors.push_back(&temperature);
      temperature_old.update_ghost_values();
      vectors.push_back(&temperature_old);
    }

    void
    distribute_constraints()
    {
      scratch_data.get_constraint(temp_dof_idx).distribute(temperature);
      scratch_data.get_constraint(temp_dof_idx).distribute(temperature_old);
    }

    void
    attach_output_vectors(DataOut<dim> &data_out) const
    {
      MeltPoolDG::VectorTools::update_ghost_values(temperature, temperature_old, heat_source);
      if (velocity)
        MeltPoolDG::VectorTools::update_ghost_values(*velocity);

      if (level_set_as_heaviside)
        MeltPoolDG::VectorTools::update_ghost_values(*level_set_as_heaviside);


      /**
       *  temperature
       */
      data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                               temperature,
                               "temperature");
      /**
       *  temperature old
       */
      data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                               temperature_old,
                               "temperature_old");
      /**
       *  heat source
       */
      data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                               heat_source,
                               "heat_source");
    }

    const VectorType &
    get_temperature() const
    {
      return temperature;
    }

    VectorType &
    get_temperature()
    {
      return temperature;
    }

    const VectorType &
    get_heat_source() const
    {
      return heat_source;
    }

    VectorType &
    get_heat_source()
    {
      return heat_source;
    }
  };
} // namespace MeltPoolDG::Heat
