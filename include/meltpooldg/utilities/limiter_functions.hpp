#pragma once

#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/limiter_data.hpp>

#include <utility>

namespace MeltPoolDG::Utilities
{
  /**
   * Computes the ratio $r = a / b$ of two slopes $a$ (`numerator`) and $b$ (`denominator`), which
   * is the input of the limiter functions $\phi(r)$ below. The limited slope is then given by
   * $b \, \phi(r)$. In the classical slope limiting setting, $a$ and $b$ are the forward and
   * backward difference slopes $u_{x,i+1/2}$ and $u_{x,i-1/2}$.
   *
   * For a vanishing denominator, the ratio is set to zero. Since all limiter functions are bounded,
   * the limited slope $b \, \phi(r)$ is zero in this case independent of the value of $r$.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param numerator The slope $a$ in the numerator of the ratio.
   * @param denominator The slope $b$ in the denominator of the ratio.
   *
   * @return The ratio $r = a / b$.
   */
  template <typename number>
  number
  compute_slope_ratio(const number &numerator, const number &denominator);

  /**
   * Computes the component-wise ratio $r = a / b$ of two slopes, see the overload above.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param numerator The slopes $a$ in the numerator of the ratio.
   * @param denominator The slopes $b$ in the denominator of the ratio.
   *
   * @return The component-wise ratio $r = a / b$.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  compute_slope_ratio(const dealii::Tensor<1, n_components, number> &numerator,
                      const dealii::Tensor<1, n_components, number> &denominator);

  /**
   * Evaluates the minmod limiter function
   * @f[
   *   \phi(r) = \max(0, \min(1, r))
   * @f]
   * for the ratio $r = a / b$ of two slopes, see compute_slope_ratio(). The limited slope
   * $b \, \phi(r)$ equals $\operatorname{minmod}(a, b)$, i.e.
   * - the slope with the smaller magnitude if both slopes share the same sign, and
   * - zero if the signs differ (indicating a local extremum), to damp oscillations.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The ratio of two slopes.
   *
   * @return The value of the limiter function.
   */
  template <typename number>
  number
  tvd_minmod(const number &r);

  /**
   * Evaluates the minmod limiter function component-wise for the ratio $r = a / b$ of two slopes,
   * see the overload above.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  tvd_minmod(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the superbee limiter function of Roe
   * @f[
   *   \phi(r) = \max(0, \min(2r, 1), \min(r, 2))
   * @f]
   * for the ratio $r = a / b$ of two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The ratio of two slopes.
   *
   * @return The value of the limiter function.
   */
  template <typename number>
  number
  superbee(const number &r);

  /**
   * Evaluates the superbee limiter function component-wise for the ratio $r = a / b$ of two
   * slopes, see the overload above.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  superbee(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the monotonized central limiter function of van Leer, commonly used in MUSCL schemes,
   * @f[
   *   \phi(r) = \max\left(0, \min\left(2r, \frac{1 + r}{2}, 2\right)\right)
   * @f]
   * for the ratio $r = a / b$ of two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The ratio of two slopes.
   *
   * @return The value of the limiter function.
   */
  template <typename number>
  number
  muscl(const number &r);

  /**
   * Evaluates the monotonized central limiter function component-wise for the ratio $r = a / b$ of
   * two slopes, see the overload above.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  muscl(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the van Leer limiter function
   * @f[
   *   \phi(r) = \frac{r + |r|}{1 + |r|}
   * @f]
   * for the ratio $r = a / b$ of two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The ratio of two slopes.
   *
   * @return The value of the limiter function.
   */
  template <typename number>
  number
  harmonic(const number &r);

  /**
   * Evaluates the van Leer limiter function component-wise for the ratio $r = a / b$ of two slopes,
   * see the overload above.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param r The component-wise ratio of two slopes.
   *
   * @return The component-wise values of the limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  harmonic(const dealii::Tensor<1, n_components, number> &r);

  /**
   * Evaluates the limiter function selected by `limiter_function_type` for the ratio $r = a / b$ of
   * two slopes, see compute_slope_ratio().
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param limiter_function_type The limiter function to evaluate.
   * @param r The ratio $r = a / b$ of two slopes.
   *
   * @return The value of the selected limiter function.
   */
  template <typename number>
  number
  evaluate_limiter_function(const LimiterFunctionType limiter_function_type, const number &r);

  /**
   * Evaluates the limiter function selected by `limiter_function_type` component-wise for the
   * ratio $r = a / b$ of two slopes, see the overload above.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param limiter_function_type The limiter function to evaluate.
   * @param r The component-wise ratio $r = a / b$ of two slopes.
   *
   * @return The component-wise values of the selected limiter function.
   */
  template <int n_components, typename number>
  dealii::Tensor<1, n_components, number>
  evaluate_limiter_function(const LimiterFunctionType                      limiter_function_type,
                            const dealii::Tensor<1, n_components, number> &r);

  /**
   * Computes the limited slopes of a piecewise linear reconstruction on the two faces of a cell in
   * one coordinate direction. With the one-sided slopes
   * @f[
   *   s_- = \frac{u - u_-}{d_-}, \qquad s_+ = \frac{u_+ - u}{d_+},
   * @f]
   * the reconstructed face values are
   * @f[
   *   u_{-1/2} = u - \frac{\Delta x}{2} \, s_+ \phi\left(\frac{s_-}{s_+}\right), \qquad
   *   u_{+1/2} = u + \frac{\Delta x}{2} \, s_- \phi\left(\frac{s_+}{s_-}\right),
   * @f]
   * where $\Delta x$ is the width of the cell. For symmetric limiter functions, i.e.
   * $\phi(1/r) = \phi(r)/r$, both slopes coincide.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param value_left The value of the neighbor in negative direction.
   * @param value The value of the cell.
   * @param value_right The value of the neighbor in positive direction.
   * @param distance_left The distance between the centers of the cell and its neighbor in negative
   * direction.
   * @param distance_right The distance between the centers of the cell and its neighbor in positive
   * direction.
   * @param limiter_function_type The limiter function to apply.
   *
   * @return The limited slopes for the face in negative and in positive direction, in this order.
   */
  template <typename number>
  std::pair<number, number>
  compute_limited_slope(const number             &value_left,
                        const number             &value,
                        const number             &value_right,
                        const number              distance_left,
                        const number              distance_right,
                        const LimiterFunctionType limiter_function_type);

  /**
   * Computes the limited slopes of a piecewise linear reconstruction on the two faces of a cell in
   * one coordinate direction component-wise, see the overload above.
   *
   * @note The number type may also be a dealii::VectorizedArray.
   *
   * @param value_left The values of the neighbor in negative direction.
   * @param value The values of the cell.
   * @param value_right The values of the neighbor in positive direction.
   * @param distance_left The distance between the centers of the cell and its neighbor in negative
   * direction.
   * @param distance_right The distance between the centers of the cell and its neighbor in positive
   * direction.
   * @param limiter_function_type The limiter function to apply.
   *
   * @return The component-wise limited slopes for the face in negative and in positive direction,
   * in this order.
   */
  template <int n_components, typename number>
  std::pair<dealii::Tensor<1, n_components, number>, dealii::Tensor<1, n_components, number>>
  compute_limited_slope(const dealii::Tensor<1, n_components, number> &value_left,
                        const dealii::Tensor<1, n_components, number> &value,
                        const dealii::Tensor<1, n_components, number> &value_right,
                        const number                                   distance_left,
                        const number                                   distance_right,
                        const LimiterFunctionType                      limiter_function_type);

} // namespace MeltPoolDG::Utilities
