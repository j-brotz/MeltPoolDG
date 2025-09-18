#include <deal.II/base/exceptions.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_wrapper.hpp>
#include <meltpooldg/time_integration/implicit_explicit_integrator.hpp>

#include <meltpooldg/utilities/matrix_type_wrapper.h>

#include <utility>

namespace MeltPoolDG::TimeIntegration
{
  template <unsigned int dim, typename number>
  ImplicitExplicitIntegrator<dim, number>::ImplicitExplicitIntegrator(
    const TimeIntegratorData<number> &time_integrator_data)
    : TimeIntegratorBase<number>(time_integrator_data)
  {}

  template <unsigned int dim, typename number>
  unsigned
  ImplicitExplicitIntegrator<dim, number>::required_solution_history_size() const
  {
    return 2;
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::reinit(const VectorType &vector_template)
  {
    intermediate_explicit_solution.reinit(vector_template, true);
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::reinit(
    const SolutionHistory<VectorType> &solution_history)
  {
    reinit(solution_history.get_current_solution());
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::configure_explicit_step(
    ExplicitRhsFunctionType explicit_rhs)
  {
    explicit_compute_rhs = std::move(explicit_rhs);
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::configure_implicit_step_wo_internal_nonlinear_solver(
    CustomSolverType custom_solver_in)
  {
    custom_solver = std::move(custom_solver_in);
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::configure_implicit_step(
    JacobianType              jacobian,
    ResidualType              residual,
    DistributeConstraintsType constraints)
  {
    compute_jacobian       = std::move(jacobian);
    compute_residual       = std::move(residual);
    distribute_constraints = std::move(constraints);

    solver.emplace(
      NewtonRaphsonSolver<number, VectorType>(this->time_integrator_data.nlsolver_data));

    // Ensure that a preconditioner is set. If not set previously it is set to identity but can be
    // changed by the user anytime by calling set_preconditioner().
    if (!preconditioner.is_initialized())
      {
        preconditioner = Preconditioner<dim, VectorType, number>(
          IdentityPreconditioner<dim, VectorType, number>());
      }
    preconditioner_update_flag = true;
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::set_preconditioner(
    Preconditioner<dim, VectorType, number> &&preconditioner_in)
  {
    preconditioner             = std::move(preconditioner_in);
    preconditioner_update_flag = true;
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::perform_time_step(
    const number                                                         current_time,
    const number                                                         time_step,
    SolutionHistory<VectorType>                                         &solution_history,
    const std::function<void(number, VectorType &, const VectorType &)> &stage_pre_processing,
    const std::function<void(number, VectorType &, const VectorType &)> &stage_post_processing)
  {
    if (stage_pre_processing)
      stage_pre_processing(current_time,
                           solution_history.get_current_solution(),
                           solution_history.get_current_solution());

    apply_explicit_step(current_time,
                        time_step,
                        solution_history.get_current_solution(),
                        intermediate_explicit_solution);

    apply_implicit_step(current_time,
                        time_step,
                        intermediate_explicit_solution,
                        solution_history.get_current_solution());



    if (stage_post_processing)
      stage_post_processing(current_time,
                            solution_history.get_current_solution(),
                            solution_history.get_current_solution());

    ++n_steps_performed;
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::apply_explicit_step(number            time,
                                                               number            time_step,
                                                               const VectorType &src,
                                                               VectorType       &dst) const
  {
    Assert(explicit_compute_rhs,
           dealii::ExcMessage(
             "No function has been set for computing the right-hand side in the explicit step!"));

    dst.zero_out_ghost_values();
    explicit_compute_rhs(
      time, time_step, dst, src, true, [&](const unsigned start_range, const unsigned end_range) {
        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = start_range; i < end_range; ++i)
          {
            dst.local_element(i) *= time_step;
            dst.local_element(i) += src.local_element(i);
          }
      });
  }

  template <unsigned int dim, typename number>
  void
  ImplicitExplicitIntegrator<dim, number>::apply_implicit_step(number      time,
                                                               number      time_step,
                                                               VectorType &explicit_solution,
                                                               VectorType &solution)
  {
    if (custom_solver)
      {
        custom_solver(time, time_step, explicit_solution, solution);
      }
    else
      {
        Assert(
          compute_residual,
          dealii::ExcMessage(
            "No function has been set for computing the computing the residual in the implicit step!"));
        Assert(
          compute_jacobian,
          dealii::ExcMessage(
            "No function has been set for computing the computing the jacobian in the implicit step!"));
        Assert(solver.has_value(), dealii::ExcInternalError());

        // update preconditioner if required
        if (n_steps_performed % this->time_integrator_data.preconditioner_update_frequency == 0 or
            preconditioner_update_flag)
          {
            preconditioner.update();
            preconditioner_update_flag = false;
          }
        solver->norm_of_solution_vector = [&solution = solution]() -> number {
          return solution.l2_norm();
        };

        // setup solver
        solver->distribute_constraints = [&](VectorType &solution) -> void {
          if (distribute_constraints)
            distribute_constraints(solution);
        };

        solver->reinit_vector = [&solution = solution](VectorType &vec) -> void {
          vec.reinit(solution);
        };

        solver->residual = [&](const VectorType &src, VectorType &dst) {
          compute_residual(time, time_step, src, dst, intermediate_explicit_solution);
        };

        std::function<void(VectorType &, const VectorType &)> jacobian_multiplication =
          [&](VectorType &dst, const VectorType &src) {
            compute_jacobian(time, time_step, dst, src);
          };

        MatrixTypeObject<VectorType> jacobian(jacobian_multiplication);

        solver->solve_with_jacobian = [&jacobian = jacobian,
                                       &data     = this->time_integrator_data.linear_solver_data,
                                       &preconditioner = preconditioner](const VectorType &rhs,
                                                                         VectorType       &dst) {
          return LinearSolver::solve(jacobian, dst, rhs, data, preconditioner);
        };

        // use explicit solution as initial guess
        solution = explicit_solution;
        solver->solve(solution);
      }
  }

  template class ImplicitExplicitIntegrator<1, double>;
  template class ImplicitExplicitIntegrator<2, double>;
  template class ImplicitExplicitIntegrator<3, double>;
} // namespace MeltPoolDG::TimeIntegration
