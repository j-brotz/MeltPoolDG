/**
 * @brief Class providing different low storage Runge-Kutta schemes. The schemes implemented in
 * this class are presented in
 *
 * Kennedy, C.A., Carpenter, M.H & Lewis, R.M. (2000). Low-storage, explicit Runge-Kutta schemes for
 * the compressible Navier-Stokes equations. Applied Numerical Mathematics, 35(2000), 177-219.
 */

#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/time_integration/explicit_integrator_base.hpp>
#include <meltpooldg/time_integration/time_integrator_data.hpp>

#include <string>
#include <variant>
#include <vector>

namespace MeltPoolDG::TimeIntegration
{
  using namespace dealii;

  template <typename number, typename PDEOperator = std::monostate>
  class LowStorageRungeKuttaIntegrator final : public ExplicitIntegratorBase<number, PDEOperator>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<number>;

    /**
     * Constructor. Set the coefficients for the low storage explicit Runge-Kutta scheme.
     *
     *  @param parameters Name of the used low-storage explicit Runge-Kutta scheme.
     *  @param timer Timer used for computation time tracking.
     */
    LowStorageRungeKuttaIntegrator(const TimeIntegratorData &parameters, TimerOutput &timer)
      : timer(timer)
    {
      switch (parameters.integrator_type)
        {
            case TimeIntegratorSchemes::LSRK_stage_3_order_3: {
              // Three-stage, third order scheme given in deal.II. While its stability region is
              // significantly smaller than for the other schemes, it only involves three stages,
              // making it very competitive in terms of the work per stage.
              bi       = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
              ai       = {{0.755726351946097, 0.386954477304099}};
              n_stages = 3;
              break;
            }
            case TimeIntegratorSchemes::LSRK_stage_5_order_4: {
              // Five-stage, fourth order scheme RK4(3)5[2R+]C by Kennedy et al.
              bi       = {{1153189308089. / 22510343858157.,
                           1772645290293. / 4653164025191.,
                           -1672844663538. / 4480602732383.,
                           2114624349019. / 3568978502595.,
                           5198255086312. / 14908931495163.}};
              ai       = {{970286171893. / 4311952581923.,
                           6584761158862. / 12103376702013.,
                           2251764453980. / 15575788980749.,
                           26877169314380. / 34165994151039.}};
              n_stages = 5;
              break;
            }
            case TimeIntegratorSchemes::LSRK_stage_7_order_4: {
              // Seven-stage, fourth order scheme given in deal.II
              bi       = {{0.0941840925477795334,
                           0.149683694803496998,
                           0.285204742060440058,
                           -0.122201846148053668,
                           0.0605151571191401122,
                           0.345986987898399296,
                           0.186627171718797670}};
              ai       = {{0.241566650129646868 + bi[0],
                           0.0423866513027719953 + bi[1],
                           0.215602732678803776 + bi[2],
                           0.232328007537583987 + bi[3],
                           0.256223412574146438 + bi[4],
                           0.0978694102142697230 + bi[5]}};
              n_stages = 7;
              break;
            }
            case TimeIntegratorSchemes::LSRK_stage_9_order_5: {
              // Nine-stage, fifth order RK5(4)9[2R+]S scheme by Kennedy et al. It is the most
              // accurate among the schemes used here, but the higher order of accuracy sacrifices
              // some stability, so the step length normalized per stage is less than for the fourth
              // order schemes.
              bi       = {{2274579626619. / 23610510767302.,
                           693987741272. / 12394497460941.,
                           -347131529483. / 15096185902911.,
                           1144057200723. / 32081666971178.,
                           1562491064753. / 11797114684756.,
                           13113619727965. / 44346030145118.,
                           393957816125. / 7825732611452.,
                           720647959663. / 6565743875477.,
                           3559252274877. / 14424734981077.}};
              ai       = {{1107026461565. / 5417078080134.,
                           38141181049399. / 41724347789894.,
                           493273079041. / 11940823631197.,
                           1851571280403. / 6147804934346.,
                           11782306865191. / 62590030070788.,
                           9452544825720. / 13648368537481.,
                           4435885630781. / 26285702406235.,
                           2357909744247. / 11371140753790.}};
              n_stages = 9;
              break;
            }
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }

      // Compute c-coefficients
      ci.reserve(n_stages);
      ci.push_back(0.0);
      number sum_previous_bi = 0.0;
      for (unsigned int stage = 1; stage < n_stages; ++stage)
        {
          ci.push_back(sum_previous_bi + ai[stage - 1]);
          sum_previous_bi += bi[stage - 1];
        }
    }

    unsigned int
    required_solution_history_size() const override
    {
      return 1;
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
      rk_register_ki.reinit(vector_template);
      rk_register_ri.reinit(vector_template);
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

    /**
     * Perform the actual time integration for a single time step.
     *
     * @param pde_operator Class providing the 'apply_operator()' function to compute f(y).
     * @param current_time Current time.
     * @param time_step Current time step size.
     * @param solution_history Solution history object providing the current and all required
     * previous solutions.
     */
    void
    perform_time_step(const PDEOperator                              &pde_operator,
                      const number                                    current_time,
                      const number                                    time_step,
                      ::TimeIntegration::SolutionHistory<VectorType> &solution_history) override
    {
      TimerOutput::Scope timer_section(timer, "Explicit Runge-Kutta time integration");
      AssertDimension(ai.size() + 1, bi.size());
      pde_operator.apply_operator(
        current_time,
        rk_register_ri,
        solution_history.get_current_solution(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (unsigned int i = start_range; i < end_range; ++i)
            {
              const number k_i   = rk_register_ri.local_element(i);
              const number sol_i = solution_history.get_current_solution().local_element(i);
              solution_history.get_current_solution().local_element(i) =
                sol_i + bi[0] * time_step * k_i;
              rk_register_ri.local_element(i) = sol_i + ai[0] * time_step * k_i;
            }
        });

      for (unsigned int stage = 1; stage < bi.size(); ++stage)
        {
          pde_operator.apply_operator(
            current_time,
            rk_register_ki,
            rk_register_ri,
            [&](const unsigned int start_range, const unsigned int end_range) {
              if (stage < bi.size() - 1)
                {
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      const number k_i   = rk_register_ki.local_element(i);
                      const number sol_i = solution_history.get_current_solution().local_element(i);
                      solution_history.get_current_solution().local_element(i) =
                        sol_i + bi[stage] * time_step * k_i;
                      rk_register_ki.local_element(i) = sol_i + ai[stage] * time_step * k_i;
                    }
                }
              else
                {
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      const number k_i   = rk_register_ki.local_element(i);
                      const number sol_i = solution_history.get_current_solution().local_element(i);
                      solution_history.get_current_solution().local_element(i) =
                        sol_i + bi[stage] * time_step * k_i;
                    }
                }
            });
          rk_register_ri.swap(rk_register_ki);
        }
    }

    inline static constexpr std::array<TimeIntegratorSchemes, 4> supported_schemes{{
      TimeIntegratorSchemes::LSRK_stage_3_order_3,
      TimeIntegratorSchemes::LSRK_stage_5_order_4,
      TimeIntegratorSchemes::LSRK_stage_7_order_4,
      TimeIntegratorSchemes::LSRK_stage_9_order_5,
    }};

  private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;

    unsigned int n_stages;

    // register for time integrator
    LinearAlgebra::distributed::Vector<number> rk_register_ri;
    LinearAlgebra::distributed::Vector<number> rk_register_ki;

    TimerOutput &timer;
  };

  // Define an alias for static access of variables
  using StaticExplicitLowStorageRungeKuttaIntegrator = LowStorageRungeKuttaIntegrator<double>;
} // namespace MeltPoolDG::TimeIntegration