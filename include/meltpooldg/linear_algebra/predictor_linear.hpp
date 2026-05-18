#pragma once

namespace MeltPoolDG
{
  /**
   * Compute a linearly extrapolated initial guess value (predictor) for the
   * Newton-Raphson solver at time "t_n+1" from the old solution @param old_vec
   * computed at time "t_n" and the old old solution @param old_old_vec computed
   * at time "t_n-1". In addition the @param current_time_increment
   * dt_n+1 = t_n+1 - t_n and the @param old_time_incrementdt_n = t_n-t_n-1 have
   * to be provided.
   *
   * The predictor is computed as follows
   *
   *                    dt_n+1
   *  q_n+1^(0) = q_n + ------ * (q_n-q_n-1)
   *                    dt_n
   *
   */
  template <typename VectorType, typename number>
  void
  compute_linear_predictor(const VectorType &old_vec,
                           const VectorType &old_old_vec,
                           VectorType       &predictor,
                           const number      current_time_increment,
                           const number      old_time_increment);
} // namespace MeltPoolDG
