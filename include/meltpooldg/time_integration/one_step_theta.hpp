#pragma once

#include <deal.II/base/config.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/time_integration/time_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>
#include <meltpooldg/utilities/solution_history.hpp>

#include <meltpooldg/utilities/matrix_type_wrapper.h>

#include <functional>
#include <ostream>

namespace MeltPoolDG
{
  using namespace dealii;

  inline static constexpr std::array<TimeIntegratorSchemes, 3> one_step_theta_supported_schemes{
    {TimeIntegratorSchemes::explicit_euler,
     TimeIntegratorSchemes::implicit_euler,
     TimeIntegratorSchemes::crank_nicolson}};

  template <typename number, typename PDEOperator>
  class OneStepTheta final : public TimeIntegratorBase<number>
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    OneStepTheta(const PDEOperator              &pde_operator,
                 const TimeIntegratorData       &time_integrator_data_in,
                 const LinearSolverData<number> &linear_solver_data_in)
      : TimeIntegratorBase<number>(time_integrator_data_in)
      , pde_operator(pde_operator)
      , linear_solver_data(linear_solver_data_in)
    {
      switch (this->time_integrator_data.integrator_type)
        {
            case TimeIntegratorSchemes::explicit_euler: {
              Theta_ = 0.0;
              break;
            }

            case TimeIntegratorSchemes::implicit_euler: {
              Theta_ = 1.0;
              break;
            }

            case TimeIntegratorSchemes::crank_nicolson: {
              Theta_ = 0.5;
              break;
            }
            default: {
              AssertThrow(
                false,
                ExcMessage(
                  "The chosen time integration scheme is not supported by the OneStepTheta class."));
            }
        }
    };

    unsigned int
    required_solution_history_size() const override
    {
      return 1;
    }

    /**
     * Performs one time step according to the one-step theta method u(t+dt) = u(t) +
     * dt*theta*F(u(t+dt)) + dt*(1-theta)*F(u(t)) So far only a linear operator is possible. This
     * results in a linear system of equations. The right hand side of System consists of quantities
     * known at time t, which are stored in rhs. Current solution of solution history is updated
     * according to the solution of the linear system.
     *
     * In addition, if @p monitoring_vector is set in the base class the right-hand-side of the
     * resulting linear equation system is copied to the monitoring vector.
     *
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     * @param pre_processing Function which is executed at the beginning before the actual time step computations.
     * Three variables are passed to the function: the current time, the vector which is later used
     * as rhs vector in the linear solver, and the current solution.
     * @param post_processing Function which is executed at the end of the time step computation. Three variables are
     * passed to the function: the current time after the time step, i.e. n+1, the solution vector
     * which is then the final solution of the time step and the current solution. Note, that these
     * to vectors are the same.
     *
     * @throw The function throws an exception if the linear solver does not converge.
     */
    void
    perform_time_step(
      const number                                                         current_time,
      const number                                                         time_step,
      ::TimeIntegration::SolutionHistory<VectorType>                      &solution_history,
      const std::function<void(number, VectorType &, const VectorType &)> &pre_processing,
      const std::function<void(number, VectorType &, const VectorType &)> &post_processing) override
    {
      Assert(solution_history.size() >= required_solution_history_size(),
             dealii::ExcMessage(
               "The size of the solution history object does not fit the requirements of the "
               "chosen time integration scheme."));
      if (pre_processing)
        pre_processing(current_time, right_hand_side_, solution_history.get_recent_old_solution());
      dt_       = time_step;
      old_time_ = current_time - time_step;
      // Create the right hand side
      right_hand_side_.reinit(solution_history.get_recent_old_solution());
      pde_operator.apply_operator(old_time_,
                                  right_hand_side_,
                                  solution_history.get_recent_old_solution(),
                                  std::function<void(unsigned int, unsigned int)>());
      right_hand_side_ *= (1.0 - Theta_);
      buffer = 0.;
      pde_operator.apply_dirichlet_boundary_operator(old_time_,
                                                     buffer,
                                                     solution_history.get_recent_old_solution());
      right_hand_side_.add(1.0 - Theta_, buffer);
      buffer = 0.;
      pde_operator.apply_dirichlet_boundary_operator(old_time_ + dt_,
                                                     buffer,
                                                     solution_history.get_recent_old_solution());
      right_hand_side_.add(Theta_, buffer);
      right_hand_side_ *= dt_;
      right_hand_side_.add(1.0, solution_history.get_recent_old_solution());

      if (this->monitoring_vector != nullptr)
        *this->monitoring_vector = right_hand_side_;

      const MatrixTypeObject<VectorType> linear_solver_matrix(
        [&](VectorType &dst, const VectorType &src) -> void {
          dst = 0.; // TODO: Initial guess
          pde_operator.apply_operator(old_time_ + dt_,
                                      dst,
                                      src,
                                      std::function<void(unsigned int, unsigned int)>());
          dst *= -1.0;
          dst *= Theta_ * dt_;
          dst.add(1.0, src);
        });


      /* TODO: add preconditioner*/
      LinearSolver::solve<VectorType>(linear_solver_matrix,
                                      solution_history.get_current_solution(),
                                      right_hand_side_,
                                      linear_solver_data);

      if (post_processing)
        post_processing(current_time + time_step,
                        solution_history.get_current_solution(),
                        solution_history.get_current_solution());
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
      buffer.reinit(vector_template);
    }

    /**
     * Sets up the necessary internal data structures by internally calling
     * reinit(solution_history.get_current_solution()).
     */
    void
    reinit(const ::TimeIntegration::SolutionHistory<VectorType> &solution_history) override
    {
      reinit(solution_history.get_current_solution());
    }


  private:
    double Theta_;

    mutable number dt_;
    mutable number old_time_;

    mutable VectorType old_solution_;
    mutable VectorType right_hand_side_;
    mutable VectorType buffer;

    const PDEOperator              &pde_operator;
    const LinearSolverData<number> &linear_solver_data;
  };
} // namespace MeltPoolDG
