/*
 * @brief Incompressible Navier-Stokes solver using the functionality of adaflo.
 */

#pragma once

#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include "meltpooldg/core/material_data.hpp"
#include "meltpooldg/core/scratch_data.hpp"
#include "meltpooldg/utilities/attach_vectors.hpp"
#include "meltpooldg/utilities/matrix_free_util.hpp"

#include <memory>
#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/base/function_parser.h>

#  include <deal.II/matrix_free/fe_evaluation.h>
#  include <deal.II/matrix_free/matrix_free.h>
#  include <deal.II/matrix_free/operators.h>

#  include <deal.II/numerics/vector_tools.h>

#  include <meltpooldg/core/simulation_base.hpp>
#  include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#  include <meltpooldg/flow/compressible_flow_boundary_conditions.hpp>
#  include <meltpooldg/flow/flow_utils.hpp>
#  include <meltpooldg/fluid_structure_interaction/fluid_structure_interaction_data.hpp>
#  include <meltpooldg/post_processing/generic_data_out.hpp>

#  include <adaflo/navier_stokes.h>

namespace MeltPoolDG::Flow
{
  template <int dim, typename number>
  class IncompressibleFlowSolverWrapper
  {
  public:
    IncompressibleFlowSolverWrapper() = default;

    IncompressibleFlowSolverWrapper(const AdafloWrapperParameters<number>       &flow_data,
                                    const FluidStructureInteractionData<number> &fsi_data,
                                    dealii::Triangulation<dim>                  &triangulation,
                                    const dealii::Mapping<dim>                  &mapping,
                                    ScratchData<dim, dim, number>               &scratch_data)
      : navier_stokes(std::make_unique<adaflo::NavierStokes<dim>>(mapping,
                                                                  flow_data.get_parameters(),
                                                                  triangulation))
      , flow_data(flow_data)
      , fsi_data(fsi_data)
      , scratch_data(&scratch_data)
    {
      dof_index_u = scratch_data.attach_dof_handler(navier_stokes->get_dof_handler_u());
      dof_index_p = scratch_data.attach_dof_handler(navier_stokes->get_dof_handler_p());
      scratch_data.attach_dof_handler(navier_stokes->get_dof_handler_u());
    }

    void
    reinit()
    {
      navier_stokes->initialize_matrix_free(
        &scratch_data->get_matrix_free(), dof_index_u, dof_index_p, quad_index_u, quad_index_p);
    }

    void
    distribute_dofs()
    {
      // navier_stokes->distribute_dofs();
    }

    void
    initialize_data_structures()
    {
      navier_stokes->distribute_dofs();

      quad_index_u = flow_data.params.use_simplex_mesh ?
                       scratch_data->attach_quadrature(
                         dealii::QGaussSimplex<dim>(flow_data.params.velocity_degree + 1)) :
                       scratch_data->attach_quadrature(
                         dealii::QGauss<dim>(flow_data.params.velocity_degree + 1));
      quad_index_p =
        flow_data.params.use_simplex_mesh ?
          scratch_data->attach_quadrature(
            dealii::QGaussSimplex<dim>(flow_data.params.velocity_degree)) :
          scratch_data->attach_quadrature(dealii::QGauss<dim>(flow_data.params.velocity_degree));

      scratch_data->attach_constraint_matrix(navier_stokes->get_constraints_u());
      scratch_data->attach_constraint_matrix(navier_stokes->get_constraints_p());
      dof_index_hanging_nodes_u =
        scratch_data->attach_constraint_matrix(navier_stokes->get_hanging_node_constraints_u());

      navier_stokes->initialize_data_structures();
    }

    /**
     * Solve the Navier-Stokes equation for a single time step. This function assumes that all
     * relevant updates for the right-hand side and others were already done before the function
     * call.
     *
     * @param dt Size of the time step
     * @param time Actual time
     */
    void
    solve(const number, const number dt)
    {
      // TODO
      update_penalty_term(dt);
      navier_stokes->get_constraints_u().set_zero(navier_stokes->user_rhs.block(0));
      navier_stokes->get_constraints_p().set_zero(navier_stokes->user_rhs.block(1));

      navier_stokes->time_stepping.set_time_step(dt);
      navier_stokes->init_time_advance(false);

      navier_stokes->solve_nonlinear_system(navier_stokes->compute_initial_residual(true));
      // distribute_constraints();

      navier_stokes->get_hanging_node_constraints_u().distribute(navier_stokes->solution.block(0));
      navier_stokes->get_hanging_node_constraints_u().distribute(
        navier_stokes->solution_old.block(0));
      navier_stokes->get_hanging_node_constraints_u().distribute(
        navier_stokes->solution_old_old.block(0));

      navier_stokes->get_hanging_node_constraints_p().distribute(navier_stokes->solution.block(1));
      navier_stokes->get_hanging_node_constraints_p().distribute(
        navier_stokes->solution_old.block(1));
      navier_stokes->get_hanging_node_constraints_p().distribute(
        navier_stokes->solution_old_old.block(1));

      navier_stokes->get_constraints_u().distribute(navier_stokes->user_rhs.block(0));
      navier_stokes->get_constraints_p().distribute(navier_stokes->user_rhs.block(1));
    }

    void
    add_external_force(
      const std::shared_ptr<Flow::IncompressibleExternalFluidForce<dim, number, dim>>
        &external_force)
    {
      external_forces.push_back(external_force);
    }

    /**
     * Prepare the Navier-Stokes solver for the coarsening and refinement of the triangulation.
     */
    void
    prepare_for_coarsening_and_refinement();

    /**
     * Execute all required Navier-Stokes solver related steps after coarsening and refinement of
     * the triangulation. This includes redistributing the DoFs and reinitializing required objects.
     */
    void
    execute_coarsening_and_refinement();


    void
    attach_for_coarsening_and_refinement(
      DoFHandlerAndVectorDataType<dim, dealii::LinearAlgebra::distributed::Vector<number>> &in)
    {
      in.emplace_back(&scratch_data->get_dof_handler(dof_index_u),
                      [&](
                        std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vec_in) {
                        vec_in.push_back(&navier_stokes->solution.block(0));
                        vec_in.push_back(&navier_stokes->solution_old.block(0));
                        vec_in.push_back(&navier_stokes->solution_old_old.block(0));
                      });

      in.emplace_back(&scratch_data->get_dof_handler(dof_index_p),
                      [&](
                        std::vector<dealii::LinearAlgebra::distributed::Vector<number> *> &vec_in) {
                        vec_in.push_back(&navier_stokes->solution.block(1));
                        vec_in.push_back(&navier_stokes->solution_old.block(1));
                        vec_in.push_back(&navier_stokes->solution_old_old.block(1));
                      });
    }

    /**
     * @brief Set the solution vector to the passed initial flow field state.
     *
     * @param function Initial condition of the flow field.
     */
    void
    set_initial_condition(const dealii::Function<dim> &function)
    {
      navier_stokes->solution.zero_out_ghost_values();
      navier_stokes->solution_old.zero_out_ghost_values();
      navier_stokes->solution_old_old.zero_out_ghost_values();
      dealii::VectorTools::interpolate(navier_stokes->mapping,
                                       navier_stokes->get_dof_handler_u(),
                                       function,
                                       navier_stokes->solution.block(0));

      // the hanging node constraints contain the inhomogeneity
      navier_stokes->get_hanging_node_constraints_u().distribute(navier_stokes->solution.block(0));
      navier_stokes->get_hanging_node_constraints_p().distribute(navier_stokes->solution.block(1));

      navier_stokes->solution.update_ghost_values();
      navier_stokes->solution_old.update_ghost_values();
      navier_stokes->solution_old_old.update_ghost_values();
    }

    /**
     * @brief Set the boundary conditions.
     *
     * @param simulation_case dealii::Pointer to the considered simulation case class.
     * @param operation_name String for the name of the considered operation.
     *
     * @note The function simply passes the parameters to the set_boundary_conditions function in the
     * CompressibleFlowBoundaryConditions class.
     */
    void
    set_boundary_conditions(const std::shared_ptr<SimulationCaseBase<dim, number>> &simulation_case,
                            const std::string                                      &operation_name)
    {
      std::map<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        boundary_conditions;

      // inflow
      boundary_conditions = simulation_case->get_boundary_condition("inflow", operation_name);
      for (const auto &[boundary_id, boundary_function] : boundary_conditions)
        navier_stokes->set_velocity_dirichlet_boundary(boundary_id, std::move(boundary_function));

      // constant pressure outflow
      boundary_conditions =
        simulation_case->get_boundary_condition("outflow_fixed_pressure", operation_name);
      for (const auto &[boundary_id, boundary_function] : boundary_conditions)
        navier_stokes->set_open_boundary_with_normal_flux(boundary_id,
                                                          std::move(boundary_function));
      // TODO
      // navier_stokes->set_open_boundary(boundary_id, std::move(boundary_function));

      // slip wall
      boundary_conditions = simulation_case->get_boundary_condition("slip_wall", operation_name);
      for (const auto &[boundary_id, boundary_function] : boundary_conditions)
        navier_stokes->set_symmetry_boundary(boundary_id);

      // no slip-wall
      boundary_conditions = simulation_case->get_boundary_condition("no_slip_wall", operation_name);
      for (const auto &[boundary_id, boundary_function] : boundary_conditions)
        navier_stokes->set_no_slip_boundary(boundary_id);
    }

    /**
     * @brief Compute the maximum time step size.
     *
     * The maximum time step size arises from the convective and viscous time step limits.
     * Optionally, it is printed to the console.
     *
     * @param do_print If true, the time step limit is printed to the console.
     *
     * @return The computed maximum time step size.
     */
    number
    compute_time_step_size() const
    {
      return 1e8;
    }

    /**
     * @brief Attach the solution to the passed data out object.
     *
     * The solution which are added are the density, the momentum and the energy density.
     *
     * @param data_out Object to which the solution vector is attached.
     */
    void
    attach_output_vectors(GenericDataOut<dim, number> &data_out) const
    {
      // pressure
      std::vector<std::string> names = {"pressure"};
      std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation = {dealii::DataComponentInterpretation::component_is_scalar};
      data_out.add_data_vector(this->navier_stokes->get_dof_handler_p(),
                               this->navier_stokes->solution.block(1),
                               names,
                               data_component_interpretation);

      // velocity
      names.clear();
      data_component_interpretation.clear();
      names.insert(names.end(), dim, "velocity");
      data_component_interpretation.insert(
        data_component_interpretation.end(),
        dim,
        dealii::DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(this->navier_stokes->get_dof_handler_u(),
                               this->navier_stokes->solution.block(0),
                               names,
                               data_component_interpretation);
    }

    // Although the solution provided by the Navier-Stokes solver is a BlockVector, we only return
    // the velocity block here.
    inline dealii::LinearAlgebra::distributed::Vector<number> &
    get_velocity_solution();

    MatrixFreeContext<dim, number>
    get_matrix_free_context_u()
    {
      return {scratch_data->get_matrix_free(), dof_index_u, quad_index_u};
    }

    MatrixFreeContext<dim, number>
    get_matrix_free_context_p()
    {
      return {scratch_data->get_matrix_free(), dof_index_p, quad_index_p};
    }

    Flow::AdafloWrapperParameters<number> &
    get_flow_params()
    {
      return flow_data;
    }

  private:
    void
    update_solution_ghost_values()
    {
      if (!navier_stokes->solution.has_ghost_elements())
        navier_stokes->solution.zero_out_ghost_values();
      navier_stokes->solution.update_ghost_values();
    }

    void
    update_penalty_term(const number time_step_size)
    {
      update_solution_ghost_values();
      dealii::LinearAlgebra::distributed::Vector<number> &navier_stokes_user_rhs =
        navier_stokes->user_rhs.block(0);
      scratch_data->initialize_dof_vector(navier_stokes_user_rhs, dof_index_u);
      navier_stokes_user_rhs = 0;


      FECellIntegrator<dim, dim, number> fe_evaluation(scratch_data->get_matrix_free(),
                                                       dof_index_u,
                                                       quad_index_u);

      for (unsigned int cell = 0; cell < navier_stokes->matrix_free->n_cell_batches(); ++cell)
        {
          fe_evaluation.reinit(cell);
          fe_evaluation.gather_evaluate(navier_stokes->solution.block(0),
                                        dealii::EvaluationFlags::values);
          dealii::VectorizedArray<number> *implicit_penalty =
            navier_stokes->get_matrix().begin_damping_coeff(cell);

          // Reset implicit penalty coefficients stored in the Navier-Stokes solver.
          std::fill(&implicit_penalty[0],
                    &implicit_penalty[fe_evaluation.n_q_points],
                    dealii::VectorizedArray<number>(0.0));

          for (auto &external_force : external_forces)
            external_force->cell_operation(
              {scratch_data->get_matrix_free(), dof_index_u, quad_index_u}, cell);

          for (unsigned int q : fe_evaluation.quadrature_point_indices())
            {
              dealii::Tensor<1, dim, dealii::VectorizedArray<number>> rhs_penalty;

              // const rhs contribution
              for (auto &external_force : external_forces)
                {
                  rhs_penalty +=
                    external_force->get_rhs_at_q(time_step_size, fe_evaluation.quadrature_point(q));
                  implicit_penalty[q] +=
                    external_force->get_damping_coeff_at_q(time_step_size,
                                                           fe_evaluation.quadrature_point(q));
                }
              fe_evaluation.submit_value(rhs_penalty, q);
            }
          fe_evaluation.integrate_scatter(dealii::EvaluationFlags::values, navier_stokes_user_rhs);
        }
      navier_stokes_user_rhs.compress(dealii::VectorOperation::add);
      navier_stokes_user_rhs.update_ghost_values();
    }

    std::unique_ptr<adaflo::NavierStokes<dim>> navier_stokes;

    AdafloWrapperParameters<number> flow_data;

    FluidStructureInteractionData<number> fsi_data;

    ScratchData<dim, dim, number> *scratch_data;

    unsigned dof_index_u, dof_index_p;

    unsigned quad_index_u, quad_index_p;

    unsigned dof_index_hanging_nodes_u;

    /// This set of pointers may hold a list of external fluid force contributions to the residuum
    /// (e.g., gravity, or user-defined source terms)
    std::vector<std::shared_ptr<Flow::IncompressibleExternalFluidForce<dim, number, dim>>>
      external_forces;
  };

  // Inline functions
  template <int dim, typename number>
  inline dealii::LinearAlgebra::distributed::Vector<number> &
  IncompressibleFlowSolverWrapper<dim, number>::get_velocity_solution()
  {
    return navier_stokes->solution.block(0);
  }

} // namespace MeltPoolDG::Flow

#endif