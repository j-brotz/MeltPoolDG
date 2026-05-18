#include <meltpooldg/heat/apparent_capacity.hpp>

namespace MeltPoolDG::Heat
{
  using namespace dealii;
  /**
   * Unit bell curve function with a polynomial degree 4
   *
   * returns f(x) = 15/16*x^4 - 15/8*x^2 + 15/16
   *
   * This curve satisfies the following statements:
   * f(1) = 0
   * f(-1) = 0
   * f'(1) = 0
   * f'(-1) = 0
   * int_{-1}^1 f(x) dx = 1
   */
  template <typename number>
  number
  unit_poly4_bell(const number x)
  {
    const number a         = 15. / 16;
    const number b         = -15. / 8;
    const number x_squared = x * x;
    return a * x_squared * x_squared + b * x_squared + a;
  }

  /**
   * Derivative of unit bell curve function unit_poly4_bell
   *
   * returns f(x) = 15/4*x^3 - 15/4*x
   */
  template <typename number>
  number
  unit_poly4_bell_derivative(const number x)
  {
    const number a = 15. / 4;
    return a * x * x * x - a * x;
  }

  /**
   * Evaluate a quadratic-linear-quadratic interpolation with zero slope at the
   * interval endpoints.
   *
   * The function constructs a piecewise curve on the interval @p x1 to @p x2.
   * Close to the endpoints, quadratic segments are used so that the derivative
   * vanishes at @p x1 and @p x2. In the center of the interval, a linear segment
   * connects both quadratic parts. The transition width is controlled by
   * @p delta.
   *
   * The resulting function satisfies:
   * - @f$ f(x_1) = y_1 @f$
   * - @f$ f(x_2) = y_2 @f$
   * - @f$ f'(x_1) = 0 @f$
   * - @f$ f'(x_2) = 0 @f$
   *
   * @tparam number Scalar type used for the interpolation.
   *
   * @param x Evaluation point.
   * @param x1 Left interval endpoint.
   * @param x2 Right interval endpoint.
   * @param y1 Function value at the left endpoint @p x1.
   * @param y2 Function value at the right endpoint @p x2.
   * @param delta Width of the quadratic transition regions near both endpoints.
   *
   * @return Interpolated function value at @p x.
   */
  template <typename number>
  number
  qlq_zero_slope(const number x,
                 const number x1,
                 const number x2,
                 const number y1,
                 const number y2,
                 const number delta)
  {
    const number xd1 = x1 + delta;
    const number xd2 = x2 - delta;

    const number den = delta + x1 - x2;

    const number a1 = (y1 - y2) / (2.0 * delta * den);
    const number b1 = -x1 * (y1 - y2) / (delta * den);
    const number c1 = (2.0 * delta * delta * y1 + 2.0 * delta * x1 * y1 - 2.0 * delta * x2 * y1 +
                       x1 * x1 * y1 - x1 * x1 * y2) /
                      (2.0 * delta * den);

    const number a2 = -a1;
    const number b2 = x2 * (y1 - y2) / (delta * den);
    const number c2 = (2.0 * delta * delta * y2 + 2.0 * delta * x1 * y2 - 2.0 * delta * x2 * y2 -
                       x2 * x2 * y1 + x2 * x2 * y2) /
                      (2.0 * delta * den);

    const number a3 = (y1 - y2) / den;
    const number b3 = (delta * y1 + delta * y2 + 2.0 * x1 * y2 - 2.0 * x2 * y1) / (2.0 * den);

    if (x < xd1)
      return a1 * x * x + b1 * x + c1;
    else if (x > xd2)
      return a2 * x * x + b2 * x + c2;
    else
      return a3 * x + b3;
  }


  /**
   * Evaluate the derivative of qlq_zero_slope().
   *
   * This function returns the derivative of the piecewise
   * quadratic-linear-quadratic interpolation constructed by qlq_zero_slope().
   * The derivative is zero at the interval endpoints @p x1 and @p x2, linear
   * in the quadratic transition regions, and constant in the central linear
   * region.
   *
   * @tparam number Scalar type used for the interpolation.
   *
   * @param x Evaluation point.
   * @param x1 Left interval endpoint.
   * @param x2 Right interval endpoint.
   * @param y1 Function value at the left endpoint @p x1.
   * @param y2 Function value at the right endpoint @p x2.
   * @param delta Width of the quadratic transition regions near both endpoints.
   *
   * @return Derivative of the interpolated function at @p x.
   */
  template <typename number>
  number
  qlq_zero_slope_derivative(const number x,
                            const number x1,
                            const number x2,
                            const number y1,
                            const number y2,
                            const number delta)
  {
    const number xd1 = x1 + delta;
    const number xd2 = x2 - delta;

    const number den = delta + x1 - x2;

    const number a1 = (y1 - y2) / (2.0 * delta * den);
    const number b1 = -x1 * (y1 - y2) / (delta * den);

    const number a2 = -a1;
    const number b2 = x2 * (y1 - y2) / (delta * den);

    const number a3 = (y1 - y2) / den;

    if (x < xd1)
      return 2.0 * a1 * x + b1;
    else if (x > xd2)
      return 2.0 * a2 * x + b2;
    else
      return a3;
  }

  template <typename number>
  ApparentCapacity<number>::ApparentCapacity(const MaterialData<number> &material_data_in)
    : material(material_data_in)
    , inv_mushy_zone_radius(2. / (material.liquidus_temperature - material.solidus_temperature))
    , peak(material.latent_heat_of_fusion /
           (material.liquidus_temperature - material.solidus_temperature))
    , apparent_capacity_type(material_data_in.apparent_capacity_type)
    , qlq(material)
  {
    AssertThrow(
      material.liquidus_temperature > material.solidus_temperature,
      ExcMessage(
        "The liquidus temperature must be greater than the solidus temperature! Abort..."));
  }

  template <typename number>
  number
  ApparentCapacity<number>::evaluate_qlq_from_solid_fraction(const number &solid_fraction) const
  {
    if (solid_fraction <= 0.0 or solid_fraction >= 1.0)
      return 0.0;

    const number &Ts = material.solidus_temperature;
    const number &Tl = material.liquidus_temperature;
    const number  T  = Tl - solid_fraction * qlq.dT;

    const number C_s = 0.0;
    const number C_l = 0.0;

    if (T <= qlq.T_peak)
      return qlq_zero_slope(T, Ts, qlq.T_peak, C_s, qlq.C_peak, qlq.delta);
    else
      return qlq_zero_slope(T, qlq.T_peak, Tl, qlq.C_peak, C_l, qlq.delta);
  }

  template <typename number>
  number
  ApparentCapacity<number>::compute_temperature_derivative_qlq_from_solid_fraction(
    const number &solid_fraction) const
  {
    if (solid_fraction <= 0.0 or solid_fraction >= 1.0)
      return 0.0;

    const number &Ts = material.solidus_temperature;
    const number &Tl = material.liquidus_temperature;
    const number  T  = Tl - solid_fraction * qlq.dT;

    const number C_s = 0.0;
    const number C_l = 0.0;

    if (T <= qlq.T_peak)
      return qlq_zero_slope_derivative(T, Ts, qlq.T_peak, C_s, qlq.C_peak, qlq.delta);
    else
      return qlq_zero_slope_derivative(T, qlq.T_peak, Tl, qlq.C_peak, C_l, qlq.delta);
  }

  template <typename number>
  VectorizedArray<number>
  ApparentCapacity<number>::evaluate(const VectorizedArray<number> &solid_fraction) const
  {
    VectorizedArray<number> value(0.0);

    if (apparent_capacity_type == ApparentCapacityType::constant)
      value = peak;
    else if (apparent_capacity_type == ApparentCapacityType::qlq)
      {
        for (unsigned int i = 0; i < solid_fraction.size(); ++i)
          value[i] = evaluate_qlq_from_solid_fraction(solid_fraction[i]);
      }
    else
      {
        const VectorizedArray<number> xi = 1.0 - 2.0 * solid_fraction;
        value = material.latent_heat_of_fusion * inv_mushy_zone_radius * unit_poly4_bell(xi);
      }
    const auto zero = VectorizedArray<number>(0.0);
    const auto one  = VectorizedArray<number>(1.0);

    // we use <= and >= bounds to also crop under and overshoots
    return compare_and_apply_mask<SIMDComparison::less_than_or_equal>(
      solid_fraction,
      zero,
      zero,
      compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
        solid_fraction, one, zero, value));
  }

  template <typename number>
  number
  ApparentCapacity<number>::evaluate(const number &solid_fraction) const
  {
    if (solid_fraction <= 0.0 or solid_fraction >= 1.0)
      return 0.0;

    if (apparent_capacity_type == ApparentCapacityType::constant)
      return peak;
    else if (apparent_capacity_type == ApparentCapacityType::qlq)
      return evaluate_qlq_from_solid_fraction(solid_fraction);
    else
      {
        const number xi = 1.0 - 2.0 * solid_fraction;
        return material.latent_heat_of_fusion * inv_mushy_zone_radius * unit_poly4_bell(xi);
      }
  }

  template <typename number>
  number
  ApparentCapacity<number>::compute_temperature_derivative(const number &solid_fraction) const
  {
    if (solid_fraction <= 0.0 or solid_fraction >= 1.0)
      return 0.0;

    if (apparent_capacity_type == ApparentCapacityType::constant)
      return 0;
    else if (apparent_capacity_type == ApparentCapacityType::qlq)
      return compute_temperature_derivative_qlq_from_solid_fraction(solid_fraction);
    else
      {
        const number xi = 1.0 - 2.0 * solid_fraction;
        return material.latent_heat_of_fusion * inv_mushy_zone_radius * inv_mushy_zone_radius *
               unit_poly4_bell_derivative(xi);
      }
  }

  template <typename number>
  VectorizedArray<number>
  ApparentCapacity<number>::compute_temperature_derivative(
    const VectorizedArray<number> &solid_fraction) const
  {
    VectorizedArray<number> value;

    if (apparent_capacity_type == ApparentCapacityType::constant)
      value = 0;
    else if (apparent_capacity_type == ApparentCapacityType::qlq)
      {
        for (unsigned int i = 0; i < solid_fraction.size(); ++i)
          value[i] = compute_temperature_derivative_qlq_from_solid_fraction(solid_fraction[i]);
      }
    else
      {
        const VectorizedArray<number> xi = 1.0 - 2.0 * solid_fraction;
        value = material.latent_heat_of_fusion * inv_mushy_zone_radius * inv_mushy_zone_radius *
                unit_poly4_bell_derivative(xi);
      }

    const auto zero = VectorizedArray<number>(0.0);
    const auto one  = VectorizedArray<number>(1.0);

    return compare_and_apply_mask<SIMDComparison::less_than_or_equal>(
      solid_fraction,
      zero,
      zero,
      compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
        solid_fraction, one, zero, value));
  }

  template class ApparentCapacity<double>;
} // namespace MeltPoolDG::Heat
