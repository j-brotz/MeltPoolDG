#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <meltpooldg/utilities/limiter_functions.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

template <typename number>
number
MeltPoolDG::Utilities::compute_slope_ratio(const number &numerator, const number &denominator)
{
  // Replace a vanishing denominator by one before dividing to avoid the division by zero
  const number safe_denominator = dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
    denominator, number(0.), number(1.), denominator);

  return dealii::compare_and_apply_mask<dealii::SIMDComparison::equal>(
    denominator, number(0.), number(0.), numerator / safe_denominator);
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::compute_slope_ratio(
  const dealii::Tensor<1, n_components, number> &numerator,
  const dealii::Tensor<1, n_components, number> &denominator)
{
  dealii::Tensor<1, n_components, number> r;
  for (unsigned int c = 0; c < n_components; ++c)
    r[c] = compute_slope_ratio(numerator[c], denominator[c]);

  return r;
}

template <typename number>
number
MeltPoolDG::Utilities::tvd_minmod(const number &r)
{
  return std::max(number(0.), std::min(number(1.), r));
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::tvd_minmod(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> phi;
  for (unsigned int c = 0; c < n_components; ++c)
    phi[c] = tvd_minmod(r[c]);

  return phi;
}

template <typename number>
number
MeltPoolDG::Utilities::superbee(const number &r)
{
  return std::max(number(0.),
                  std::max(std::min(number(2.) * r, number(1.)), std::min(r, number(2.))));
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::superbee(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> phi;
  for (unsigned int c = 0; c < n_components; ++c)
    phi[c] = superbee(r[c]);

  return phi;
}

template <typename number>
number
MeltPoolDG::Utilities::muscl(const number &r)
{
  return std::max(number(0.),
                  std::min(number(2.) * r, std::min(number(0.5) * (r + number(1.)), number(2.))));
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::muscl(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> phi;
  for (unsigned int c = 0; c < n_components; ++c)
    phi[c] = muscl(r[c]);

  return phi;
}

template <typename number>
number
MeltPoolDG::Utilities::harmonic(const number &r)
{
  return (r + std::abs(r)) / (number(1.) + std::abs(r));
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::harmonic(const dealii::Tensor<1, n_components, number> &r)
{
  dealii::Tensor<1, n_components, number> phi;
  for (unsigned int c = 0; c < n_components; ++c)
    phi[c] = harmonic(r[c]);

  return phi;
}

template <typename number>
number
MeltPoolDG::Utilities::evaluate_limiter_function(const LimiterFunctionType limiter_function_type,
                                                 const number             &r)
{
  switch (limiter_function_type)
    {
      case LimiterFunctionType::tvd_minmod:
        return tvd_minmod(r);
      case LimiterFunctionType::superbee:
        return superbee(r);
      case LimiterFunctionType::muscl:
        return muscl(r);
      case LimiterFunctionType::harmonic:
        return harmonic(r);
      default:
        AssertThrow(false, dealii::ExcMessage("Unknown limiter function type."));
    }
  return number(0.);
}

template <int n_components, typename number>
dealii::Tensor<1, n_components, number>
MeltPoolDG::Utilities::evaluate_limiter_function(const LimiterFunctionType limiter_function_type,
                                                 const dealii::Tensor<1, n_components, number> &r)
{
  switch (limiter_function_type)
    {
      case LimiterFunctionType::tvd_minmod:
        return tvd_minmod(r);
      case LimiterFunctionType::superbee:
        return superbee(r);
      case LimiterFunctionType::muscl:
        return muscl(r);
      case LimiterFunctionType::harmonic:
        return harmonic(r);
      default:
        AssertThrow(false, dealii::ExcMessage("Unknown limiter function type."));
    }
  return dealii::Tensor<1, n_components, number>();
}

template <typename number>
std::pair<number, number>
MeltPoolDG::Utilities::compute_limited_slope(const number             &value_left,
                                             const number             &value,
                                             const number             &value_right,
                                             const number              distance_left,
                                             const number              distance_right,
                                             const LimiterFunctionType limiter_function_type)
{
  const number slope_left  = (value - value_left) / distance_left;
  const number slope_right = (value_right - value) / distance_right;

  const number phi_left =
    evaluate_limiter_function(limiter_function_type, compute_slope_ratio(slope_left, slope_right));
  const number phi_right =
    evaluate_limiter_function(limiter_function_type, compute_slope_ratio(slope_right, slope_left));

  return std::make_pair(slope_right * phi_left, slope_left * phi_right);
}

template <int n_components, typename number>
std::pair<dealii::Tensor<1, n_components, number>, dealii::Tensor<1, n_components, number>>
MeltPoolDG::Utilities::compute_limited_slope(
  const dealii::Tensor<1, n_components, number> &value_left,
  const dealii::Tensor<1, n_components, number> &value,
  const dealii::Tensor<1, n_components, number> &value_right,
  const number                                   distance_left,
  const number                                   distance_right,
  const LimiterFunctionType                      limiter_function_type)
{
  const dealii::Tensor<1, n_components, number> slope_left = (value - value_left) / distance_left;
  const dealii::Tensor<1, n_components, number> slope_right =
    (value_right - value) / distance_right;

  const dealii::Tensor<1, n_components, number> phi_left =
    evaluate_limiter_function(limiter_function_type, compute_slope_ratio(slope_left, slope_right));
  const dealii::Tensor<1, n_components, number> phi_right =
    evaluate_limiter_function(limiter_function_type, compute_slope_ratio(slope_right, slope_left));

  return std::make_pair(dealii::schur_product(slope_right, phi_left),
                        dealii::schur_product(slope_left, phi_right));
}
