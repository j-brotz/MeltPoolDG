#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/grid/tria.h>

#include <algorithm>
#include <cmath>
#include <ios>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace MeltPoolDG::UtilityFunctions
{
  template <typename number1, typename number2>
  void
  remove_duplicates(std::vector<number1> &a, std::vector<number2> &b)
  {
    std::unordered_set<int> seen;
    auto                    it_a = a.begin();
    auto                    it_b = b.begin();

    while (it_a != a.end())
      {
        if (seen.contains(*it_a))
          {
            // If element is duplicate, remove it from both vectors
            it_a = a.erase(it_a);
            it_b = b.erase(it_b);
          }
        else
          {
            // If element is not duplicate, mark it as seen and move iterators forward
            seen.insert(*it_a);
            ++it_a;
            ++it_b;
          }
      }
  }

  template <typename T>
  std::string
  to_string_with_precision(const T a_value, const int n = 6)
  {
    std::ostringstream out;
    out.precision(n);
    out << std::scientific << a_value;
    return out.str();
  }

  template <typename number>
  inline dealii::VectorizedArray<number>
  limit_to_bounds(const dealii::VectorizedArray<number> &in,
                  const number                           lower_limit,
                  const number                           upper_limit)
  {
    const auto ub = compare_and_apply_mask<dealii::SIMDComparison::greater_than>(in,
                                                                                 upper_limit,
                                                                                 upper_limit,
                                                                                 in);
    return compare_and_apply_mask<dealii::SIMDComparison::less_than>(ub,
                                                                     lower_limit,
                                                                     lower_limit,
                                                                     ub);
  }

  template <typename number>
  inline number
  limit_to_bounds(const number in, const number lower_limit, const number upper_limit)
  {
    const auto ub = in > upper_limit ? upper_limit : in;
    return ub < lower_limit ? lower_limit : ub;
  }

  /**
   * This function returns 1.0 if the \p in value is between (excluding) the \p lower_limit and
   * \p upper_limit. Otherwise, this function returns 0.0.
   */
  template <typename number>
  dealii::VectorizedArray<number>
  is_between(const dealii::VectorizedArray<number> &in,
             const number                           lower_limit,
             const number                           upper_limit)
  {
    return compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      in,
      upper_limit,
      compare_and_apply_mask<dealii::SIMDComparison::greater_than>(in, lower_limit, 1.0, 0.0),
      0.0);
  }

  /**
   * Return the exponent to the power of ten of an expression like 5*10^5 --> return 5
   */
  template <typename number>
  inline int
  get_exponent_power_ten(const number x)
  {
    if (x >= 1e-16) // positive number
      return std::floor(std::log10(x));
    else if (x <= 1e-16) // negative number
      return std::floor(std::log10(std::abs(x)));
    else // number close to 0
      return 0;
  }

  /*
   * 1d numerical integration of @p vals given at @p points using a trapezoidal rule.
   */
  template <int dim, typename number>
  number
  integrate_over_line(const std::vector<number>             &vals,
                      const std::vector<dealii::Point<dim>> &points)
  {
    number result = 0;
    for (unsigned int i = 0; i < vals.size() - 1; ++i)
      result += (vals[i] + vals[i + 1]) / 2. * (points[i + 1].distance(points[i]));

    return result;
  }

  template <int dim, typename number>
  number
  compute_numerical_zero_of_norm(const dealii::Triangulation<dim> &triangulation,
                                 const dealii::Mapping<dim>       &mapping)
  {
    return std::min(1e-2,
                    std::max(std::pow(10,
                                      UtilityFunctions::get_exponent_power_ten(std::pow(
                                        dealii::GridTools::volume<dim>(triangulation, mapping),
                                        1. / dim))) *
                               1e-3,
                             1e-12));
  }

  template <int dim, typename ForwardIterator>
  dealii::Point<dim>
  to_point(const ForwardIterator begin, const ForwardIterator end)
  {
    (void)end;
    AssertIndexRange(dim, std::distance(begin, end) + 1);

    dealii::Point<dim> point;

    auto it = begin;
    for (int i = 0; i < dim; ++i)
      point[i] = *it++;

    return point;
  }

  /**
   * Helper function for the computation of a weighted average:
   * weight_a * term_a + weight_b * term_b
   */
  template <typename TypeWeight, typename TypeTerm>
  TypeTerm
  calculate_arithmetic_phase_weighted_average(const TypeWeight &weight_a,
                                              const TypeTerm   &term_a,
                                              const TypeWeight &weight_b,
                                              const TypeTerm   &term_b)
  {
    return weight_a * term_a + weight_b * term_b;
  }
} // namespace MeltPoolDG::UtilityFunctions
