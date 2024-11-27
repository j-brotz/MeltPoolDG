#pragma once

/* Code adapted from:
https://github.com/kronbichler/advection_miniapp/blob/master/advection_solver_variable.cc*/

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/operators.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/linear_algebra/linear_solver_data.hpp>
#include <meltpooldg/time_integration/time_integration_base.hpp>
#include <meltpooldg/time_integration/time_integration_setup.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <iostream>

/**
 *  Runge-Kutta time integrator schemes
 */
namespace MeltPoolDG
{
  using namespace dealii;

  template <typename Operator, int dim, TimeIntegrators scheme, typename Number = double>
  class LowStorageRungeKuttaIntegrator : public TimeIntegrationBase<dim>
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

    LowStorageRungeKuttaIntegrator(Operator                                       &pde_operator,
                                   const MeltPoolDG::ScratchData<dim>             &scratch_data_in,
                                   const unsigned int                              dof_idx_in,
                                   const unsigned int                              quad_idx_in,
                                   [[maybe_unused]] const LinearSolverData<Number> linear_solver)
      : pde_operator_(pde_operator)
      , scratch_data_(scratch_data_in)
      , dof_idx(dof_idx_in)
      , quad_idx(quad_idx_in)
    {
      AssertThrow(scratch_data_.is_FE_DGQ(dof_idx), ExcMessage("This works only for DG elements."));
      switch (scheme)
        {
            case TimeIntegrators::RK_stage_1_order_1: /*Forward Euler*/
            {
              bi = {{1.0}};
              ai = {{}};
              break;
            }

            case TimeIntegrators::RK_stage_2_order_2: /*Heuns Method*/
            {
              bi = {{0.5, 0.5}};
              ai = {{1.0}};
              break;
            }

            case TimeIntegrators::RK_stage_3_order_3: {
              bi = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
              ai = {{0.755726351946097, 0.386954477304099}};
              break;
            }
            case TimeIntegrators::RK_stage_5_order_4: {
              bi = {{1153189308089. / 22510343858157.,
                     1772645290293. / 4653164025191.,
                     -1672844663538. / 4480602732383.,
                     2114624349019. / 3568978502595.,
                     5198255086312. / 14908931495163.}};
              ai = {{970286171893. / 4311952581923.,
                     6584761158862. / 12103376702013.,
                     2251764453980. / 15575788980749.,
                     26877169314380. / 34165994151039.}};

              break;
            }

            case TimeIntegrators::RK_stage_7_order_4: {
              bi = {{0.0941840925477795334,
                     0.149683694803496998,
                     0.285204742060440058,
                     -0.122201846148053668,
                     0.0605151571191401122,
                     0.345986987898399296,
                     0.186627171718797670}};
              ai = {{0.241566650129646868 + bi[0],
                     0.0423866513027719953 + bi[1],
                     0.215602732678803776 + bi[2],
                     0.232328007537583987 + bi[3],
                     0.256223412574146438 + bi[4],
                     0.0978694102142697230 + bi[5]}};
              break;
            }
            case TimeIntegrators::RK_stage_9_order_5: {
              bi = {{2274579626619. / 23610510767302.,
                     693987741272. / 12394497460941.,
                     -347131529483. / 15096185902911.,
                     1144057200723. / 32081666971178.,
                     1562491064753. / 11797114684756.,
                     13113619727965. / 44346030145118.,
                     393957816125. / 7825732611452.,
                     720647959663. / 6565743875477.,
                     3559252274877. / 14424734981077.}};
              ai = {{1107026461565. / 5417078080134.,
                     38141181049399. / 41724347789894.,
                     493273079041. / 11940823631197.,
                     1851571280403. / 6147804934346.,
                     11782306865191. / 62590030070788.,
                     9452544825720. / 13648368537481.,
                     4435885630781. / 26285702406235.,
                     2357909744247. / 11371140753790.}};
              break;
            }


          default:
            AssertThrow(false, ExcNotImplemented());
        }

      // AssertThrow(scratch_data_.is_FE_DGQ(dof_idx), ExcMessage("This works only for DG
      // elements."));
    }

    /**
     * Perfoms a time step following a low storage runge kutta scheme. See  Kennedy, Carpenter,
     * Lewis, 2000 for example. ri and ki are vectors that are used to store intermediate
     * results. The current solution of the solution history is updated, based on the old solution.
     * rhs is not needed here, but is required to be passed to keep a consistent interface for
     * different time integrators.
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
      AssertDimension(ai.size() + 1, bi.size());

      const bool update_ghost_values = solution_history.get_current_solution().has_ghost_elements();
      if (update_ghost_values)
        {
          solution_history.get_current_solution().update_ghost_values();
        }

      ri = solution_history.get_current_solution();

      perform_stage(old_time,
                    bi[0] * time_step,
                    ai[0] * time_step,
                    solution_history.get_current_solution());



      Number sum_previous_bi = 0;

      for (unsigned int stage = 1; stage < bi.size(); ++stage)
        {
          const Number c_i = sum_previous_bi + ai[stage - 1];
          perform_stage(old_time + c_i * time_step,
                        bi[stage] * time_step,
                        (stage == bi.size() - 1 ? 0 : ai[stage] * time_step),
                        solution_history.get_current_solution());
          sum_previous_bi += bi[stage - 1];
        }

      if (update_ghost_values)
        solution_history.get_current_solution().zero_out_ghost_values();
    }

    /**
     * This function updates the size the vectors ri and ki. It also reinitilizes the pde underlying
     * pde operator.
     */
    void
    reinit() override
    {
      this->scratch_data_.get_matrix_free().initialize_dof_vector(this->ri, dof_idx);
      this->scratch_data_.get_matrix_free().initialize_dof_vector(this->ki, dof_idx);
    }

  private:
    // Coefficients for the Runge-Kutta scheme.
    std::vector<Number> bi;
    std::vector<Number> ai;

    Operator          &pde_operator_;
    mutable VectorType ri;
    mutable VectorType ki;

    const MeltPoolDG::ScratchData<dim> &scratch_data_;

    const unsigned int dof_idx  = 0;
    const unsigned int quad_idx = 0;

    /* TODO: extend to CG*/
    void
    local_apply_inverse_mass_matrix(const MatrixFree<dim, Number>               &data,
                                    VectorType                                  &dst,
                                    const VectorType                            &src,
                                    const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FECellIntegrator<dim, 1, Number> eval(data, dof_idx, quad_idx);


      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1, Number> inverse(eval);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          eval.reinit(cell);
          eval.read_dof_values(src);

          inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

          eval.set_dof_values(dst);
        }
    }

    /** This function performs a single stage of the Runge-Kutta scheme. the solution is updated
     *based on the intermediate results and the factor_solution and factor_ai.
     *@param time
     *@param factor_solution multiplication factor for the solution
     *@param factor_ai multiplication factor for the intermediate solutions
     *@param solution solution vector
     */
    void
    perform_stage(const Number time,
                  const Number factor_solution,
                  const Number factor_ai,
                  VectorType  &solution) const
    {
      pde_operator_.apply_operator(time, this->ki, this->ri);
      pde_operator_.apply_dirichlet_boundary_operator(time, this->ki, this->ri);
      {
        ri.zero_out_ghost_values();
        scratch_data_.get_matrix_free().cell_loop(
          &LowStorageRungeKuttaIntegrator::local_apply_inverse_mass_matrix,
          this,
          ri,
          ki,
          std::function<void(const unsigned int, const unsigned int)>(),
          [&](const unsigned int start_range, const unsigned int end_range) {
            const Number ai = factor_ai;
            const Number bi = factor_solution;
            if (ai == Number())
              {
                DEAL_II_OPENMP_SIMD_PRAGMA
                for (unsigned int i = start_range; i < end_range; ++i)
                  {
                    const Number k_i          = ri.local_element(i);
                    const Number sol_i        = solution.local_element(i);
                    solution.local_element(i) = sol_i + bi * k_i;
                  }
              }
            else
              {
                DEAL_II_OPENMP_SIMD_PRAGMA
                for (unsigned int i = start_range; i < end_range; ++i)
                  {
                    const Number k_i          = ri.local_element(i);
                    const Number sol_i        = solution.local_element(i);
                    solution.local_element(i) = sol_i + bi * k_i;
                    ri.local_element(i)       = sol_i + ai * k_i;
                  }
              }
          });
      }
    }
  };
} // namespace MeltPoolDG