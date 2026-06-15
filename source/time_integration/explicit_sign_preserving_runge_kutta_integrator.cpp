#include <deal.II/base/exceptions.h>

#include <meltpooldg/time_integration/explicit_sign_preserving_runge_kutta_integrator.hpp>

#include <utility>

namespace MeltPoolDG::TimeIntegration
{
  template <typename number>
  ExplicitSignPreservingRungeKuttaIntegrator<number>::ExplicitSignPreservingRungeKuttaIntegrator(
    const TimeIntegratorData<number> &time_integrator_data)
    : TimeIntegratorBase<number>(time_integrator_data)
  {
    alpha.resize(3);
    alpha[0].resize(1);
    alpha[1].resize(2);
    alpha[2].resize(3);
    alpha[0][0] = 1.0;
    alpha[1][0] = 7. / 8.;
    alpha[1][1] = 1. / 8.;
    alpha[2][0] = 0.5;
    alpha[2][1] = 1. / 6.;
    alpha[2][2] = 1. / 3.;

    beta.resize(3);
    beta[0].resize(1);
    beta[1].resize(2);
    beta[2].resize(3);
    beta[0][0] = 2. / 3.;
    beta[1][0] = 1. / 12.;
    beta[1][1] = 1. / 2.;
    beta[2][0] = 1. / 12.;
    beta[2][1] = 1. / 12.;
    beta[2][2] = 1. / 2.;
  }

  template <typename number>
  unsigned
  ExplicitSignPreservingRungeKuttaIntegrator<number>::required_solution_history_size() const
  {
    return 1;
  }

  template <typename number>
  void
  ExplicitSignPreservingRungeKuttaIntegrator<number>::configure_rhs(
    const RhsFunctionType &compute_rhs_in)
  {
    compute_rhs = std::move(compute_rhs_in);
  }


  template <typename number>
  void
  ExplicitSignPreservingRungeKuttaIntegrator<number>::reinit(const VectorType &vector_template)
  {
    w_1.reinit(vector_template);
    w_2.reinit(vector_template);
    w_new.reinit(vector_template);
    w_helper.reinit(vector_template);
  }


  template <typename number>
  void
  ExplicitSignPreservingRungeKuttaIntegrator<number>::reinit(
    const SolutionHistory<VectorType> &solution_history)
  {
    reinit(solution_history.get_current_solution());
  }


  template <typename number>
  void
  ExplicitSignPreservingRungeKuttaIntegrator<number>::perform_time_step(
    const number                 current_time,
    const number                 time_step,
    SolutionHistory<VectorType> &solution_history,
    const std::function<void(number, number, VectorType &, const VectorType &)>
      &stage_pre_processing,
    const std::function<void(number, number, VectorType &, const VectorType &)>
      &stage_post_processing)
  {
    Assert(solution_history.size() >= required_solution_history_size(),
           dealii::ExcMessage(
             "The size of the solution history object does not fit the requirements of the "
             "chosen time integration scheme."));

    const number inverse_a_1 = 1. / alpha[0][0];
    const number inverse_a_2 = 1. / (alpha[1][0] + alpha[1][1]);
    const number inverse_a_3 = 1. / (alpha[2][0] + alpha[2][1] + alpha[2][2]);

    // Step 1
    if (stage_pre_processing)
      stage_pre_processing(current_time,
                           time_step,
                           solution_history.get_current_solution(),
                           solution_history.get_current_solution());
    w_helper = 0;
    compute_rhs(current_time,
                time_step,
                w_helper,
                solution_history.get_current_solution(),
                [&](const unsigned int start_range, const unsigned int end_range) {
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      w_1.local_element(i) =
                        inverse_a_1 *
                        (alpha[0][0] * solution_history.get_current_solution().local_element(i) +
                         beta[0][0] * w_helper.local_element(i));
                      w_2.local_element(i) =
                        inverse_a_2 *
                        (alpha[1][0] * solution_history.get_current_solution().local_element(i) +
                         beta[1][0] * w_helper.local_element(i));
                      w_new.local_element(i) =
                        inverse_a_3 *
                        (alpha[2][0] * solution_history.get_current_solution().local_element(i) +
                         beta[2][0] * w_helper.local_element(i));
                    }
                });

    if (stage_post_processing)
      stage_post_processing(current_time, time_step, w_1, w_1);

    // Step 2
    w_helper = 0;
    compute_rhs(current_time,
                time_step,
                w_helper,
                w_1,
                [&](const unsigned int start_range, const unsigned int end_range) {
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      w_2.local_element(i) +=
                        inverse_a_2 * (alpha[1][1] * w_1.local_element(i) +
                                       beta[1][1] * w_helper.local_element(i));
                      w_new.local_element(i) +=
                        inverse_a_3 * (alpha[2][1] * w_1.local_element(i) +
                                       beta[2][1] * w_helper.local_element(i));
                    }
                });

    if (stage_post_processing)
      stage_post_processing(current_time, time_step, w_2, w_2);

    // Stage 3
    w_helper = 0;
    compute_rhs(current_time,
                time_step,
                w_helper,
                w_1,
                [&](const unsigned int start_range, const unsigned int end_range) {
                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i = start_range; i < end_range; ++i)
                    {
                      solution_history.get_current_solution().local_element(i) =
                        w_new.local_element(i) +
                        inverse_a_3 * (alpha[2][2] * w_2.local_element(i) +
                                       beta[2][2] * w_helper.local_element(i));
                    }
                });

    if (stage_post_processing)
      stage_post_processing(current_time,
                            time_step,
                            solution_history.get_current_solution(),
                            solution_history.get_current_solution());
  }

  template class ExplicitSignPreservingRungeKuttaIntegrator<double>;
} // namespace MeltPoolDG::TimeIntegration
