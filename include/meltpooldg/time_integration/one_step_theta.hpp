#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/tools.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/time_integration/time_integration_base.hpp>
#include <meltpooldg/time_integration/time_integration_setup.hpp>

#include <ostream>



namespace MeltPoolDG
{
  using namespace dealii;

  template <typename Operator, int dim, TimeIntegrators scheme, typename Number = double>
  class OneStepTheta : public TimeIntegrationBase<dim>
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    OneStepTheta(Operator                                        &pde_operator,
                 const MeltPoolDG::ScratchData<dim>              &scratch_data_in,
                 const unsigned int                               dof_idx_in,
                 const unsigned int                               quad_idx_in,
                 [[maybe_unused]] const LinearSolverData<Number> &linear_solver_data_in)
      : pde_operator_(pde_operator)
      , scratch_data_(scratch_data_in)
      , dof_idx(dof_idx_in)
      , quad_idx(quad_idx_in)
    {
      AssertThrow(scratch_data_.is_FE_DGQ(dof_idx), ExcMessage("This works only for DG elements."));
      switch (scheme)
        {
            case TimeIntegrators::explicit_euler: {
              Theta_ = 0.0;
              break;
            }

            case TimeIntegrators::implicit_euler: {
              Theta_ = 1.0;
              break;
            }

            case TimeIntegrators::crank_nicolson: {
              Theta_ = 0.5;
              break;
            }
            default: {
              // AssertThrow(false, ExcMessage("This works only for DG elements."));
            }
        }

      linear_solver_data = linear_solver_data_in;
    };

    /**
     * Performs one time step according to the one step theta method u(t+dt) = u(t) +
     * dt*theta*F(u(t+dt)) + dt*(1-theta)*F(u(t)) So far only a linear operator is possible. This
     * results in a linear system of equations. The right hand side of System consists of quantities
     * known at time t, which are stored in rhs. Current solution of solution history is updated
     * according to the solution of the linear system.
     * @param old_time
     * @param time_step size of the current time step
     * @param solution_history object holding the advection field at different time instances.
     * @param rhs right hand side vector for implicit time integration schemes
     */
    void
    perform_time_step(
      [[maybe_unused]] const double                                  old_time,
      [[maybe_unused]] const double                                 &time_step,
      [[maybe_unused]] TimeIntegration::SolutionHistory<VectorType> &solution_history,
      [[maybe_unused]] VectorType                                   &rhs) const override
    {
      dt_       = time_step;
      old_time_ = old_time;

      create_right_hand_side(solution_history.get_recent_old_solution());

      /* TODO: add preconditioner*/
      LinearSolver::solve<VectorType>(*this,
                                      solution_history.get_current_solution(),
                                      right_hand_side_,
                                      linear_solver_data);
    }


    /**
     *  performs a matrix vector multiplication in matrix free implementation.
     * @param src source vector of the matrix multiplication
     * @param dst result of the matrix multiplication
     * */
    void
    vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      pde_operator_.apply_operator(old_time_ + dt_, dst, src, true);

      dst *= -1.0;
      dst *= Theta_ * dt_;
      dst.add(1.0, src);
    }

    /**
     *  Allocates memory for the vectors based on the degrees of freedom of the DoFHandler.
     * */
    void
    reinit() override
    {
      this->scratch_data_.get_matrix_free().initialize_dof_vector(this->buffer, dof_idx);
    };

  private:
    /**
     * Fills rhs with values known at time t. Optionally the vector can be zeroed out.
     * @param zero_out flag if the right hand side should be zeroed out before.
     * @param old_solution
     */
    void
    create_right_hand_side(const VectorType &old_solution) const
    {
      right_hand_side_.reinit(old_solution);

      pde_operator_.apply_operator(old_time_, right_hand_side_, old_solution, true);

      right_hand_side_ *= (1.0 - Theta_);

      pde_operator_.apply_dirichlet_boundary_operator(old_time_, buffer, old_solution);
      right_hand_side_.add(1.0 - Theta_, buffer);

      pde_operator_.apply_dirichlet_boundary_operator(old_time_ + dt_, buffer, old_solution);
      right_hand_side_.add(Theta_, buffer);

      right_hand_side_ *= dt_;

      right_hand_side_.add(1.0, old_solution);
    }


    /* TODO: extend to CG*/
    void
    local_apply_inverse_mass_matrix(const MatrixFree<dim, Number>                    &data,
                                    LinearAlgebra::distributed::Vector<Number>       &dst,
                                    const LinearAlgebra::distributed::Vector<Number> &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FECellIntegrator<dim, 1, Number> eval(data, 0, 0);

      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, Number> inverse(eval);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval.reinit(cell);
          eval.read_dof_values(src);

          inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

          eval.set_dof_values(dst);
        }
    }

    Operator &pde_operator_;

    double Theta_;

    mutable double dt_;
    mutable double old_time_;

    mutable VectorType old_solution_;
    mutable VectorType right_hand_side_;
    mutable VectorType buffer;


    const MeltPoolDG::ScratchData<dim> &scratch_data_;

    const unsigned int dof_idx;
    const unsigned int quad_idx;

    LinearSolverData<Number> linear_solver_data;
  };
} // namespace MeltPoolDG
