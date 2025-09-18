#include <deal.II/base/exceptions.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner_trilinos_wrapper.hpp>
#include <meltpooldg/time_integration/bdf_time_integration.hpp>

#include <meltpooldg/utilities/matrix_type_wrapper.h>

#include <utility>

namespace MeltPoolDG::TimeIntegration
{
  template <int dim, typename number>
  BDFIntegrator<dim, number>::BDFIntegrator(const TimeIntegratorData<number> &time_integrator_data)
    : TimeIntegratorBase<number>(time_integrator_data)
    , solver(std::make_unique<NewtonRaphsonSolver<number, VectorType>>(
        this->time_integrator_data.nlsolver_data))
  {}

  template <int dim, typename number>
  void
  BDFIntegrator<dim, number>::configure_solver_functions(JacobianType              jacobian,
                                                         ResidualType              residual,
                                                         DistributeConstraintsType constraints)
  {
    compute_jacobian       = std::move(jacobian);
    compute_residual       = std::move(residual);
    distribute_constraints = std::move(constraints);

    // Ensure that a preconditioner is set. If not set previously it is set to identity but can be
    // changed by the user anytime by calling set_preconditioner().
    if (!preconditioner.is_initialized())
      {
        preconditioner = Preconditioner<dim, VectorType, number>(
          IdentityPreconditioner<dim, VectorType, number>());
      }
    preconditioner_update_flag = true;
  }

  template <int dim, typename number>
  void
  BDFIntegrator<dim, number>::set_preconditioner(
    Preconditioner<dim, VectorType, number> &&preconditioner_in)
  {
    preconditioner             = std::move(preconditioner_in);
    preconditioner_update_flag = true;
  }

  template <int dim, typename number>
  unsigned
  BDFIntegrator<dim, number>::required_solution_history_size() const
  {
    switch (this->time_integrator_data.integrator_type)
      {
        case TimeIntegratorSchemes::bdf_1:
          return 2;
        case TimeIntegratorSchemes::bdf_2:
          return 3;
        case TimeIntegratorSchemes::bdf_3:
          return 4;
        case TimeIntegratorSchemes::bdf_4:
          return 5;
        case TimeIntegratorSchemes::bdf_5:
          return 6;
        case TimeIntegratorSchemes::bdf_6:
          return 7;
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
      }
  }

  template <int dim, typename number>
  void
  BDFIntegrator<dim, number>::reinit(const VectorType &vector_template)
  {
    summed_old_solution.reinit(vector_template, true);
    preconditioner_update_flag = true;
  }

  template <int dim, typename number>
  void
  BDFIntegrator<dim, number>::reinit(const SolutionHistory<VectorType> &solution_history)
  {
    reinit(solution_history.get_current_solution());
  }

  template <int dim, typename number>
  void
  BDFIntegrator<dim, number>::perform_time_step(
    const number                                                         current_time,
    const number                                                         time_step,
    SolutionHistory<VectorType>                                         &solution_history,
    const std::function<void(number, VectorType &, const VectorType &)> &stage_pre_processing,
    const std::function<void(number, VectorType &, const VectorType &)> &stage_post_processing)
  {
    Assert(compute_jacobian and compute_residual,
           dealii::ExcMessage("The integrator has not been initialized!"));

    // Perform the initial steps with a bdf scheme of lower order, i.e. 1st step: BDF1, 2nd
    // step BDF2 and so on until the desired BDF scheme is reached
    static unsigned step_count = 0;
    if (step_count != required_solution_history_size() - 1)
      {
        ++step_count;
        set_up_bdf_parameters(step_count);
      }

    Assert(solution_history.size() > step_count,
           dealii::ExcMessage(
             "The size of the solution history object does not fit the requirements of the "
             "chosen time integration scheme."));

    compute_weighted_old_solution_sum(solution_history);

    if (stage_pre_processing)
      stage_pre_processing(current_time,
                           solution_history.get_current_solution(),
                           solution_history.get_current_solution());

    if (n_steps_performed % this->time_integrator_data.preconditioner_update_frequency == 0 or
        preconditioner_update_flag)
      {
        preconditioner.update();
        preconditioner_update_flag = false;
      }

    // matrix type wrapper for the jacobian
    std::function<void(VectorType &, const VectorType &)> jacobian_multiplication =
      [&](VectorType &dst, const VectorType &src) {
        // We pass time_step*bdf_weights.rhs as time step as the right hand side f(y) is not
        // scaled and therefore the complete equation is divided by this factor.
        compute_jacobian(time_step * bdf_weights.rhs, dst, src);
      };

    MatrixTypeObject<VectorType> jacobian(jacobian_multiplication);

    solver->norm_of_solution_vector = [&solution =
                                         solution_history.get_current_solution()]() -> number {
      return solution.l2_norm();
    };

    solver->distribute_constraints = [&](VectorType &solution) -> void {
      if (distribute_constraints)
        distribute_constraints(solution);
    };

    solver->reinit_vector = [&solution = solution_history.get_current_solution()](
                              VectorType &vec) -> void { vec.reinit(solution); };

    solver->residual = [&](const VectorType &src, VectorType &dst) {
      // We pass time_step*bdf_weights.rhs as time step as the right hand side f(y) is not scaled
      // and therefore the complete equation is divided by this factor.
      compute_residual(
        current_time + time_step, time_step * bdf_weights.rhs, src, dst, summed_old_solution);
    };

    solver->solve_with_jacobian = [&jacobian       = jacobian,
                                   &data           = this->time_integrator_data.linear_solver_data,
                                   &preconditioner = preconditioner](const VectorType &rhs,
                                                                     VectorType       &dst) {
      return LinearSolver::solve(jacobian, dst, rhs, data, preconditioner);
    };

    solver->solve(solution_history.get_current_solution());

    if (stage_post_processing)
      stage_post_processing(current_time + time_step,
                            solution_history.get_current_solution(),
                            solution_history.get_current_solution());

    ++n_steps_performed;
  }

  template <int dim, typename number>
  void
  BDFIntegrator<dim, number>::compute_weighted_old_solution_sum(
    SolutionHistory<VectorType> &solution_history)
  {
    DEAL_II_OPENMP_SIMD_PRAGMA
    for (unsigned int i = 0; i < summed_old_solution.locally_owned_size(); ++i)
      {
        summed_old_solution.local_element(i) = 0.;
        for (unsigned int j = bdf_weights.old_solutions.size(); j > 0; --j)
          {
            summed_old_solution.local_element(i) +=
              bdf_weights.old_solutions[j - 1] * solution_history.get_solution(j).local_element(i);
          }
      }
  }

  template <int dim, typename number>
  void
  BDFIntegrator<dim, number>::set_up_bdf_parameters(const unsigned int bdf_scheme)
  {
    bdf_weights.old_solutions.clear();
    switch (bdf_scheme)
      {
          case 1: {
            bdf_weights.rhs = 1.;
            bdf_weights.old_solutions.reserve(1);
            bdf_weights.old_solutions.push_back(1.);
            break;
          }
          case 2: {
            bdf_weights.rhs = 2. / 3.;
            bdf_weights.old_solutions.reserve(2);
            bdf_weights.old_solutions.push_back(4. / 3.);
            bdf_weights.old_solutions.push_back(-1. / 3.);
            break;
          }
          case 3: {
            bdf_weights.rhs = 6. / 11.;
            bdf_weights.old_solutions.reserve(3);
            bdf_weights.old_solutions.push_back(18. / 11.);
            bdf_weights.old_solutions.push_back(-9. / 11.);
            bdf_weights.old_solutions.push_back(2. / 11.);
            break;
          }
          case 4: {
            bdf_weights.rhs = 12. / 25.;
            bdf_weights.old_solutions.reserve(4);
            bdf_weights.old_solutions.push_back(48. / 25.);
            bdf_weights.old_solutions.push_back(-36. / 25.);
            bdf_weights.old_solutions.push_back(16. / 25.);
            bdf_weights.old_solutions.push_back(-3. / 25.);
            break;
          }
          case 5: {
            bdf_weights.rhs = 60. / 137.;
            bdf_weights.old_solutions.reserve(5);
            bdf_weights.old_solutions.push_back(300. / 137.);
            bdf_weights.old_solutions.push_back(-300. / 137.);
            bdf_weights.old_solutions.push_back(200. / 137.);
            bdf_weights.old_solutions.push_back(-75. / 137.);
            bdf_weights.old_solutions.push_back(12. / 137.);
            break;
          }
          case 6: {
            bdf_weights.rhs = 60. / 147.;
            bdf_weights.old_solutions.reserve(6);
            bdf_weights.old_solutions.push_back(360. / 147.);
            bdf_weights.old_solutions.push_back(-450. / 147.);
            bdf_weights.old_solutions.push_back(400. / 147.);
            bdf_weights.old_solutions.push_back(-225. / 147.);
            bdf_weights.old_solutions.push_back(72. / 147.);
            bdf_weights.old_solutions.push_back(-10. / 147.);
            break;
          }
        default:
          Assert(false, dealii::ExcMessage("This code should not be reachable!"));
      }
  }

  template class BDFIntegrator<1, double>;
  template class BDFIntegrator<2, double>;
  template class BDFIntegrator<3, double>;
} // namespace MeltPoolDG::TimeIntegration
