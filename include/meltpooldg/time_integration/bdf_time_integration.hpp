#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/newton_raphson_solver.hpp>
#include <meltpooldg/linear_algebra/preconditioner.hpp>
#include <meltpooldg/linear_algebra/preconditioner_factory.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/preprocessor_directives.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

#include <meltpooldg/utilities/matrix_type_wrapper.h>

#include <functional>
#include <string>
#include <vector>

namespace MeltPoolDG
{
  /**
   * Requirements for a pde operator in order to use it with the bdf time integrator.
   */
  template <typename Operator, typename Number, typename VectorType>
  concept BDFImplicitPDEOperator = requires(Operator          pde_operator,
                                            VectorType       &vec,
                                            const VectorType &const_vec,
                                            bool              boolean,
                                            Number            number) {
    /**
     * Given an ODE y'=f(y) this function shall compute the residual y'-f(y) = 0. The function
     * takes six parameters: current time, current y and the destination vector in
     * which f(y) shall be stored.
     */
    pde_operator.compute_residual(number, const_vec, vec);

    /**
     * Compute the result of J*x, where J is the Jacobian of the residual and x is a vector passed
     * to the function. The function arguments are: The vector x, and the vector in which the
     * result shall be stored.
     */
    pde_operator.apply_jacobian(vec, const_vec);

    /**
     * This function is intended to allow for setting class constants of the PDEOperator (e.g. the
     * current time step size) which are required by compute_residual or compute_jacobian but cannot
     * be set during the function call. The following parameters are passed to the function: current
     * time, current time step size, old solution vector used to approximate any occuring temporal
     * time derivative by (t^(n+1)-t^(n))/dt and a scaling factor such that residual and jacobian
     * are computed for an ode of the form y' = a*f(y) where 'a' is the scaling factor.
     */
    pde_operator.set_stage_constants(number, number, const_vec, number);
  };

  /**
   * A concept which need to be fullifilled by the operator if after each nonlinear iteration
   * additional constraints must be applied to the solution.
   */
  template <typename Operator, typename VectorType>
  concept SolutionRequiresConstraints = requires(Operator pde_operator, VectorType &vec) {
    /**
     * Distribute additional pde specific constraints to the given vector. The passed vector is the
     * current solution vector.
     */
    pde_operator.distribute_constraints(vec);
  };

  /**
   * The time integrator schemes supported by the bdf time integrator.
   */
  inline static constexpr std::array<TimeIntegratorSchemes, 6> bdf_supported_schemes{
    {TimeIntegratorSchemes::bdf_1,
     TimeIntegratorSchemes::bdf_2,
     TimeIntegratorSchemes::bdf_3,
     TimeIntegratorSchemes::bdf_4,
     TimeIntegratorSchemes::bdf_5,
     TimeIntegratorSchemes::bdf_6}};

  template <
    int dim,
    typename number,
    BDFImplicitPDEOperator<number, dealii::LinearAlgebra::distributed::Vector<number>> PDEOperator>
  class BDFIntegrator final : public TimeIntegratorBase<number>
  {
  public:
    using VectorType = dealii::LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor. Set the coefficients for the BDF time integration scheme.
     *
     * @param pde_operator Operator providing the necessary functionality to perform the time
     * integration.
     * @param time_integrator_data Time integrator data struct setting the scheme of the integrator.
     * @param scratch_data_in Scratch data object used in the pde operator.
     * @param dof_idx_in Relevant dof index of the passed operator in the scratch data object.
     */
    BDFIntegrator(const PDEOperator                   &pde_operator,
                  const TimeIntegratorData<number>    &time_integrator_data,
                  const ScratchData<dim, dim, number> &scratch_data_in,
                  const unsigned int                   dof_idx_in)
      : TimeIntegratorBase<number>(time_integrator_data)
      , pde_operator(pde_operator)
      , nonlinear_solver(this->time_integrator_data.nlsolver_data)
      , scratch_data(scratch_data_in)
      , dof_idx(dof_idx_in)
    {
      preconditioner = make_preconditioner<dim, number, PDEOperator, VectorType>(
        time_integrator_data.linear_solver_data.preconditioner_type, &pde_operator);
    }

    unsigned int
    required_solution_history_size() const override
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

    /**
     * Allocate memory for the required vectors used during the integration. This function needs to
     * be called once before the function perform_time_step() can be called.
     *
     * @param vector_template Reference vector used to define the partitioning for all internal
     * vectors.
     */
    void
    reinit(const VectorType &vector_template) override
    {
      summed_old_solution.reinit(vector_template, true);
      preconditioner.reinit(scratch_data, dof_idx);
    }

    /**
     * Sets up the necessary internal data structures by internally calling
     * reinit(solution_history.get_current_solution()).
     */
    void
    reinit(const ::TimeIntegration::SolutionHistory<VectorType, number> &solution_history) override
    {
      reinit(solution_history.get_current_solution());
    }

    /**
     * Perform the actual time integration for a single time step using the low storage explicit
     * Runge-Kutta scheme.
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     * @param stage_pre_processing Function which is executed before the bdf update step.
     * @param stage_post_processing Function which is executed after the bdf update step.
     * Runge-Kutta stage.
     */
    void
    perform_time_step(
      const number                                                         current_time,
      const number                                                         time_step,
      ::TimeIntegration::SolutionHistory<VectorType, number>              &solution_history,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_pre_processing,
      const std::function<void(number, VectorType &, const VectorType &)> &stage_post_processing)
      override
    {
      // Perform the initial steps with a bdf scheme of lower order, i.e. 1st step: BDF1, 2nd
      // step BDF2 and so on until the desired BDF scheme is reached
      static unsigned int step_count = 0;
      if (step_count != required_solution_history_size() - 1)
        {
          ++step_count;
          set_up_bdf_parameters(step_count);
        }

      Assert(solution_history.size() > step_count,
             dealii::ExcMessage(
               "The size of the solution history object does not fit the requirements of the "
               "chosen time integration scheme."));

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < summed_old_solution.locally_owned_size(); ++i)
        {
          summed_old_solution.local_element(i) = 0.;
          for (unsigned int j = step_count; j > 0; --j)
            {
              summed_old_solution.local_element(i) +=
                bdf_prefactors_old_solutions[j - 1] *
                solution_history.get_solution(j).local_element(i);
            }
        }

      // matrix type wrapper for the jacobian
      std::function<void(VectorType &, const VectorType &)> jacobian_multiplication =
        MPDG_LAMBDA_WRAPPER(pde_operator.apply_jacobian);
      MatrixTypeObject<VectorType> jacobian(jacobian_multiplication);

      nonlinear_solver.norm_of_solution_vector =
        [&solution = solution_history.get_current_solution()]() -> number {
        return solution.l2_norm();
      };

      nonlinear_solver.distribute_constraints = [&pde_operator =
                                                   pde_operator](VectorType &solution) -> void {
        if constexpr (SolutionRequiresConstraints<PDEOperator, VectorType>)
          pde_operator.distribute_constraints(solution);
      };

      nonlinear_solver.reinit_vector = [&solution = solution_history.get_current_solution()](
                                         VectorType &vec) -> void { vec.reinit(solution); };

      nonlinear_solver.residual =
        [&current_time, &pde_operator = pde_operator](const VectorType &src, VectorType &dst) {
          pde_operator.compute_residual(current_time, src, dst);
        };

      nonlinear_solver.solve_with_jacobian =
        [&jacobian       = jacobian,
         &data           = this->time_integrator_data.linear_solver_data,
         &preconditioner = preconditioner](const VectorType &rhs, VectorType &dst) {
          return LinearSolver::solve(jacobian, dst, rhs, data, preconditioner);
        };

      static int count = 0;
      if (count % this->time_integrator_data.preconditioner_update_frequency == 0)
        preconditioner.update();
      count++;

      if (stage_pre_processing)
        stage_pre_processing(current_time,
                             solution_history.get_current_solution(),
                             solution_history.get_current_solution());

      pde_operator.set_stage_constants(current_time + time_step,
                                       time_step,
                                       summed_old_solution,
                                       rhs_prefactor);

      nonlinear_solver.solve(solution_history.get_current_solution());

      if (stage_post_processing)
        stage_post_processing(current_time + time_step,
                              solution_history.get_current_solution(),
                              solution_history.get_current_solution());
    }

  private:
    void
    set_up_bdf_parameters(const unsigned int bdf_scheme)
    {
      bdf_prefactors_old_solutions.clear();
      switch (bdf_scheme)
        {
            case 1: {
              rhs_prefactor = 1.;
              bdf_prefactors_old_solutions.reserve(1);
              bdf_prefactors_old_solutions.push_back(1.);
              break;
            }
            case 2: {
              rhs_prefactor = 2. / 3.;
              bdf_prefactors_old_solutions.reserve(2);
              bdf_prefactors_old_solutions.push_back(4. / 3.);
              bdf_prefactors_old_solutions.push_back(-1. / 3.);
              break;
            }
            case 3: {
              rhs_prefactor = 6. / 11.;
              bdf_prefactors_old_solutions.reserve(3);
              bdf_prefactors_old_solutions.push_back(18. / 11.);
              bdf_prefactors_old_solutions.push_back(-9. / 11.);
              bdf_prefactors_old_solutions.push_back(2. / 11.);
              break;
            }
            case 4: {
              rhs_prefactor = 12. / 25.;
              bdf_prefactors_old_solutions.reserve(4);
              bdf_prefactors_old_solutions.push_back(48. / 25.);
              bdf_prefactors_old_solutions.push_back(-36. / 25.);
              bdf_prefactors_old_solutions.push_back(16. / 25.);
              bdf_prefactors_old_solutions.push_back(-3. / 25.);
              break;
            }
            case 5: {
              rhs_prefactor = 60. / 137.;
              bdf_prefactors_old_solutions.reserve(5);
              bdf_prefactors_old_solutions.push_back(300. / 137.);
              bdf_prefactors_old_solutions.push_back(-300. / 137.);
              bdf_prefactors_old_solutions.push_back(200. / 137.);
              bdf_prefactors_old_solutions.push_back(-75. / 137.);
              bdf_prefactors_old_solutions.push_back(12. / 137.);
              break;
            }
            case 6: {
              rhs_prefactor = 60. / 147.;
              bdf_prefactors_old_solutions.reserve(6);
              bdf_prefactors_old_solutions.push_back(360. / 147.);
              bdf_prefactors_old_solutions.push_back(-450. / 147.);
              bdf_prefactors_old_solutions.push_back(400. / 147.);
              bdf_prefactors_old_solutions.push_back(-225. / 147.);
              bdf_prefactors_old_solutions.push_back(72. / 147.);
              bdf_prefactors_old_solutions.push_back(-10. / 147.);
              break;
            }
          default:
            Assert(false, dealii::ExcMessage("This code should not be reachable!"));
        }
    }

    //! Operator use duing the time integration
    const PDEOperator &pde_operator;

    //! Preconditioner for the linear solver
    Preconditioner<dim, VectorType, number> preconditioner;

    //! Nonlinear solver
    NewtonRaphsonSolver<number, VectorType> nonlinear_solver;

    //! BDF prefactor for the right-hand side
    number rhs_prefactor;

    //! Prefactors of the old solutions used in the BDF scheme
    std::vector<number> bdf_prefactors_old_solutions;

    //! Sum of old solution with prefactors from BDF method
    VectorType summed_old_solution;

    const ScratchData<dim, dim, number> &scratch_data;

    const unsigned int dof_idx;
  };
} // namespace MeltPoolDG