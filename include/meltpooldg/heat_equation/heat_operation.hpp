/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/heat_equation/heat_operator.hpp>
#include <meltpooldg/utilities/newton_raphson_solver.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::HeatEquation
{
  using namespace dealii;

  template <int dim>
  class HeatOperation
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

    std::shared_ptr<HeatOperator<dim>> heat_operator;

  public:
    HeatOperation(const ScratchData<dim> &scratch_data_in,
                  const HeatData<double> &heat_data_in,
                  const unsigned int      temp_dof_idx_in,
                  const unsigned int      temp_hanging_nodes_dof_idx_in,
                  const unsigned int      temp_quad_idx_in,
                  const std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
                    &neumann_bc_in, //@todo find a nice way to provide BC
                  const std::vector<types::boundary_id> bc_radiation_in,
                  const std::vector<types::boundary_id> bc_convection_in)
      : scratch_data(scratch_data_in)
      , heat_data(heat_data_in)
      , temp_dof_idx(temp_dof_idx_in)
      , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
      , temp_quad_idx(temp_quad_idx_in)
    {
      heat_operator = std::make_shared<HeatOperator<dim>>(scratch_data,
                                                          heat_data,
                                                          neumann_bc_in,
                                                          bc_radiation_in,
                                                          bc_convection_in,
                                                          temp_dof_idx,
                                                          temp_quad_idx,
                                                          temperature,
                                                          heat_source);
    }

    void
    set_initial_condition(const VectorType &initial_temperature_field)
    {
      reinit();

      temperature.copy_locally_owned_data_from(initial_temperature_field);
      temperature_old.copy_locally_owned_data_from(initial_temperature_field);
    }

    void
    reinit()
    {
      scratch_data.initialize_dof_vector(temperature, temp_dof_idx);
      scratch_data.initialize_dof_vector(temperature_old, temp_dof_idx);
      scratch_data.initialize_dof_vector(heat_source, temp_dof_idx);
    }

    bool
    is_converged(const int         n_nonlinear_iter,
                 const VectorType &residual,
                 const VectorType &solution_update)
    {
      const double res_norm    = residual.l2_norm();
      const double update_norm = solution_update.l2_norm();

      scratch_data.get_pcout() << std::setw(15) << std::setprecision(10) << res_norm << " "
                               << std::setw(15) << std::setprecision(10) << update_norm
                               << std::endl;
      if (n_nonlinear_iter <= heat_data.nlsolve.max_nonlinear_iterations)
        {
          return (update_norm <= heat_data.nlsolve.field_correction_tolerance) &&
                 (res_norm <= heat_data.nlsolve.residual_tolerance);
        }
      else if (n_nonlinear_iter <= heat_data.nlsolve.max_nonlinear_iterations +
                                     heat_data.nlsolve.max_nonlinear_iterations_alt)
        {
          return (update_norm <= heat_data.nlsolve.field_correction_tolerance_alt) &&
                 (res_norm <= heat_data.nlsolve.residual_tolerance_alt);
        }
      else
        return false;
    }

    void
    solve(const double dt)
    {
      if (!heat_data.do_matrix_free)
        AssertThrow(false, ExcNotImplemented());

      heat_operator->set_time_increment(dt);

      const auto create_rhs = [&](VectorType &rhs) {
        heat_operator->create_rhs_and_apply_dirichlet_mf(
          rhs, temperature_old, scratch_data, temp_dof_idx, temp_hanging_nodes_dof_idx);
      };

      const auto solve_linear_system = [&](VectorType &      solution_update,
                                           const VectorType &rhs) -> int {
        return LinearSolve<VectorType, SolverGMRES<VectorType>, OperatorBase<double>>::solve(
          *heat_operator, solution_update, rhs, heat_data.solver.rel_tolerance_rhs);
      };

      auto newton = NewtonRaphsonSolver<dim>(scratch_data,
                                             heat_data.nlsolve,
                                             temp_dof_idx,
                                             temperature_old,
                                             temperature,
                                             create_rhs,
                                             solve_linear_system);

      newton.solve();

      temperature_old = temperature;
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
      MeltPoolDG::VectorTools::update_ghost_values(temperature, heat_source);
      /**
       *  temperature
       */
      data_out.add_data_vector(scratch_data.get_dof_handler(temp_dof_idx),
                               temperature,
                               "temperature");
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
} // namespace MeltPoolDG::HeatEquation
