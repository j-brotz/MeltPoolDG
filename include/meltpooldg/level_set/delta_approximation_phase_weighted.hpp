#pragma once
#include <deal.II/base/vectorization.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>

namespace MeltPoolDG::LevelSet
{
  using namespace dealii;

  template <typename number>
  class DeltaApproximationBase
  {
  public:
    virtual number
    compute_weight(const number ls_heaviside) const = 0;

    virtual VectorizedArray<number>
    compute_weight(const VectorizedArray<number> &ls_heaviside) const = 0;
  };


  /**
   * Dirac delta approximation function with a restriction towards the predominantely
   * heavy material phase:
   *
   *            /
   *            | 2 * δ(φ)   if φ > 0.5
   * δ_w(φ) =   |
   *            | 0          else.
   *            \
   *
   * The function is dependent only on the heaviside representation of the level set φ (=indicator).
   * The symmetric delta function δ(φ) is defined as the norm of the indicator gradient ∇φ:
   *
   * δ = ||∇φ||
   */
  template <typename number>
  class DeltaApproximationHeavyPhaseOnly : public DeltaApproximationBase<number>
  {
  public:
    DeltaApproximationHeavyPhaseOnly() = default;

    inline number
    compute_weight(const number level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

    inline VectorizedArray<number>
    compute_weight(const VectorizedArray<number> &level_set_heaviside) const override
    {
      VectorizedArray<number> temp(0.0);
      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        temp = compute_weight_internal(level_set_heaviside[v]);
      return temp;
    }

  private:
    template <typename value_type>
    inline value_type
    compute_weight_internal(const value_type &level_set_heaviside) const
    {
      if (level_set_heaviside > 0.5)
        return 2.0;
      else
        return 0.0;
    }
  };


  /**
   * Asymmetric, phase weighted Dirac delta approximation function, as proposed in
   *
   * Yokoi, K. (2014). A density-scaled continuum surface force model within a balanced
   * force formulation. Journal of Computational Physics, 278, 221-228.
   * DOI: 10.1016/j.jcp.2014.08.034
   *
   * This function can be used to approximate the Dirac delta function for diffuse interfaces. The
   * approximation is asymmetric and you can weigh the two phases individually. The function is
   * defined as
   *
   * δ_w(φ) = δ(φ) * W(φ).
   *
   * The function is dependent only on the heaviside representation of the level set φ (=indicator)
   * The symmetric delta function δ(φ) is defined as the norm of the indicator gradient ∇φ:
   *
   * δ = ||∇φ||
   *
   * The weight function W(φ) is defined as
   *
   *         2 ( (1 - φ) w_g + φ w_h )
   * W(φ) = --------------------------- ,
   *                w_g + w_h
   *
   * where w_g is the weight of the gaseous phase (at level set = -1) and w_h is the weight of the
   * heavy phase (at level set = 1). The weights can be chosen arbitrarily, as long as they don't
   * sum up to zero.
   *
   */
  template <typename number>
  class DeltaApproximationHeavisidePhaseWeighted : public DeltaApproximationBase<number>
  {
  public:
    DeltaApproximationHeavisidePhaseWeighted(
      const DeltaApproximationPhaseWeightedData<number> &data)
      : w_g(data.gas_phase_weight)
      , w_h(data.heavy_phase_weight)
      , correction_factor(2. / (w_g + w_h))
    {
      AssertThrow(w_g + w_h != 0.0,
                  ExcMessage("When using a phase weighted Dirac delta function approximation"
                             "use weights that don't sum up to zero! Abort..."));
    }

    inline number
    compute_weight(const number level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

    inline VectorizedArray<number>
    compute_weight(const VectorizedArray<number> &level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

  private:
    /**
     * This function calculates the asymmetric
     *
     *         2 ( (1 - φ) w_g + φ w_h )
     * W(φ) = ---------------------------
     *                w_g + w_h
     *
     */
    template <typename value_type>
    inline value_type
    compute_weight_internal(const value_type &level_set_heaviside) const
    {
      return Tools::interpolate(level_set_heaviside, w_g, w_h) * correction_factor;
    }

    const number w_g;
    const number w_h;

    /*
     * correction factor
     *
     *      2
     * -----------
     *  w_g + w_h
     */
    const number correction_factor;
  };

  /**
   * Asymmetric, double phase weighted Dirac delta approximation function.
   *
   * This function can be used to approximate the Dirac delta function for diffuse interfaces. The
   * approximation is asymmetric and you can choose two weights for each phase individually. The
   * function is defined as
   *
   * δ_w(φ) = δ(φ) * W(φ).
   *
   * with the heaviside representation of the level set φ (=indicator).
   * The symmetric delta function δ(φ) is defined as
   *
   * δ = ||∇φ||.
   *
   * The weight function W(φ) is defined as
   *
   *         6 ( (1 - φ) w_1g + φ w_2h ) ( (1 - φ) w_2g + φ w_2h )
   * W(φ) = ------------------------------------------------------- ,
   *           2 w_1g w_2g + w_1g w_2h + w_1h w_2g + 2 w_1h w_2h
   *
   * where w_1g and w_2g are the weights of the gaseous phase (at level set = -1) and w_1h and w_2h
   * are the weights of the heavy phase (at level set = 1). The weights can be chosen arbitrarily,
   * as long as 2 w_1g w_2g + w_1g w_2h + w_1h w_2g + 2 w_1h w_2h != 0.
   *
   */
  template <typename number>
  class DeltaApproximationHeavisideTimesHeavisidePhaseWeighted
    : public DeltaApproximationBase<number>
  {
  public:
    DeltaApproximationHeavisideTimesHeavisidePhaseWeighted(
      const DeltaApproximationPhaseWeightedData<number> &data)
      : w_1g(data.gas_phase_weight)
      , w_1h(data.heavy_phase_weight)
      , w_2g(data.gas_phase_weight_2)
      , w_2h(data.heavy_phase_weight_2)
      , correction_factor(6. / (2. * w_1g * w_2g + w_1g * w_2h + w_1h * w_2g + 2. * w_1h * w_2h))
    {
      AssertThrow(std::abs(2. * w_1g * w_2g + w_1g * w_2h + w_1h * w_2g + 2. * w_1h * w_2h) >
                    std::numeric_limits<number>::epsilon(),
                  ExcMessage("When using a phase quadratic weighted Dirac delta function"
                             "approximation use weights that fulfill this condition! Abort..."));
    }

    inline number
    compute_weight(const number level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

    inline VectorizedArray<number>
    compute_weight(const VectorizedArray<number> &level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

  private:
    /**
     * This function calculates the quadratic asymmetric weight function
     *
     *         6 ( (1 - φ) w_1g + φ w_2h ) ( (1 - φ) w_2g + φ w_2h )
     * W(φ) = -------------------------------------------------------
     *           2 w_1g w_2g + w_1g w_2h + w_1h w_2g + 2 w_1h w_2h
     *
     */
    template <typename value_type>
    inline value_type
    compute_weight_internal(const value_type &level_set_heaviside) const
    {
      const auto w_1 = Tools::interpolate(level_set_heaviside, w_1g, w_1h);
      const auto w_2 = Tools::interpolate(level_set_heaviside, w_2g, w_2h);
      return w_1 * w_2 * correction_factor;
    }

    const number w_1g;
    const number w_1h;
    const number w_2g;
    const number w_2h;

    /*
     * correction factor
     *
     *                          6
     * ---------------------------------------------------
     *  2 w_1g w_2g + w_1g w_2h + w_1h w_2g + 2 w_1h w_2h
     */
    const number correction_factor;
  };

  /**
   * Asymmetric, Dirac delta approximation function, phase weighted with the density distribution
   * consistent with the mass flux due to evaporation
   *
   * This function can be used to approximate the Dirac delta function for diffuse interfaces. The
   * approximation is asymmetric and you can weigh the two phases individually. The function is
   * defined as
   *
   * δ_w(φ) = δ(φ) * W(φ).
   *
   * The function is dependent only on the heaviside representation of the level set φ (=indicator)
   * The symmetric delta function δ(φ) is defined as the norm of the indicator gradient ∇φ:
   *
   * δ = ||∇φ||
   *
   * The weight function W(φ) is defined as
   *
   *                    1                       w_g - w_h
   * W(φ) = ------------------------- * -------------------------
   *         φ / w_h + (1 - φ) / w_g     w_g w_h ln( w_g / w_h )
   *
   * where w_g is the weight of the gaseous phase (at level set = -1) and w_h is the weight of the
   * heavy phase (at level set = 1). The weights can be chosen arbitrarily, as long as
   *
   * w_g > 0  and  w_h > 0  and  w_g != w_h
   *
   * @note If the density is determined consistent with mass flux due to evaporation, this Dirac
   *       delta approximation can be used to scale interface quantities with the density
   *       distribution by choosing the phases' weights equal to their densities.
   */
  template <typename number>
  class DeltaApproximationReciprocalPhaseWeighted : public DeltaApproximationBase<number>
  {
  public:
    DeltaApproximationReciprocalPhaseWeighted(
      const DeltaApproximationPhaseWeightedData<number> &data)
      : w_g(data.gas_phase_weight)
      , w_h(data.heavy_phase_weight)
      , correction_factor((w_g - w_h) / (w_g * w_h * std::log(w_g / w_h)))
    {
      AssertThrow(w_g > 0.0,
                  ExcMessage("When using the Dirac delta function approximation weighted"
                             "consistent with evaporation use positive weights! Abort..."));
      AssertThrow(w_h > 0.0,
                  ExcMessage("When using the Dirac delta function approximation weighted"
                             "consistent with evaporation use positive weights! Abort..."));
      AssertThrow(std::abs(w_g - w_h) > std::numeric_limits<number>::epsilon(),
                  ExcMessage("When using the Dirac delta function approximation weighted consistent"
                             "with evaporation the phase weights must differ! Abort..."));
    }

    inline number
    compute_weight(const number level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

    inline VectorizedArray<number>
    compute_weight(const VectorizedArray<number> &level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

  private:
    /**
     * This function calculates the asymmetric
     *
     *                    1                       w_g - w_h
     * W(φ) = ------------------------- * -------------------------
     *         φ / w_h + (1 - φ) / w_g     w_g w_h ln( w_g / w_h )
     *
     */
    template <typename value_type>
    inline value_type
    compute_weight_internal(const value_type &level_set_heaviside) const
    {
      return Tools::interpolate_reciprocal(level_set_heaviside, w_g, w_h) * correction_factor;
    }

    const number w_g;
    const number w_h;

    /*
     * correction factor
     *
     *         w_g - w_h
     * -------------------------
     *  w_g w_h ln( w_g / w_h )
     */
    const number correction_factor;
  };

  /**
   * Asymmetric, Dirac delta approximation function, phase weighted with the density distribution
   * consistent with the mass flux due to evaporation time an additional weight
   *
   * This function can be used to approximate the Dirac delta function for diffuse interfaces. The
   * approximation is asymmetric and you can weigh the two phases individually. The function is
   * defined as
   *
   * δ_w(φ) = δ(φ) * W(φ).
   *
   * The function is dependent only on the heaviside representation of the level set φ (=indicator)
   * The symmetric delta function δ(φ) is defined as the norm of the indicator gradient ∇φ:
   *
   * δ = ||∇φ||
   *
   * The weight function W(φ) is defined as
   *
   *           (1 - φ) w_2g + φ w_2h
   * W(φ) = --------------------------- * c_corr,
   *         φ / w_1h + (1 - φ) / w_1g
   *
   * with the correction factor
   *
   *           /      w_1g w_1h ln( w_1g / w_1h )
   * c_corr = | w_2g -----------------------------
   *          \              w_1g - w_1h
   *
   *                             1 / w_1g ( ln( w_1h / w_1g ) - 1 ) + 1 / w_1h  \-1
   *          + (w_2h - w_2g) * ----------------------------------------------- |
   *                                         ( 1/w_1h - 1/w_1g )²              /
   *
   * where w_1g and w_2g are the weights of the gaseous phase (at level set = -1) and w_1h and w_2h
   * are the weights of the heavy phase (at level set = 1). The weights can be chosen arbitrarily,
   * as long as
   *
   * w_1g > 0
   * w_1h > 0
   * w_1g != w_1h
   * w_2g > 0  or  w_2h != w_2g
   *
   * @note If the density is determined consistent with mass flux due to evaporation, this Dirac
   *       delta approximation can be used to scale interface quantities with the density
   *       distribution by choosing the first set of weights (w_1g and w_1h) equal to their
   *       densities. The second set of weights (w_2g and w_2h) can be chosen independently.
   */
  template <typename number>
  class DeltaApproximationReciprocalTimesHeavisidePhaseWeighted
    : public DeltaApproximationBase<number>
  {
  public:
    DeltaApproximationReciprocalTimesHeavisidePhaseWeighted(
      const DeltaApproximationPhaseWeightedData<number> &data)
      : w_1g(data.gas_phase_weight)
      , w_1h(data.heavy_phase_weight)
      , w_2g(data.gas_phase_weight_2)
      , w_2h(data.heavy_phase_weight_2)
      , correction_factor(
          // clang-format off
          1. / (
              w_2g * (w_1g * w_1h * std::log(w_1g / w_1h))
              / (w_1g - w_1h)
            +
              (w_2h - w_2g) * (1. / w_1g * (std::log(w_1h / w_1g) - 1.) + 1. / w_1h)
              / Utilities::fixed_power<2>(1. / w_1h - 1. / w_1g)
          )
          // clang-format on
        )
    {
      AssertThrow(w_1g > 0.0,
                  ExcMessage(
                    "When using the Dirac delta function approximation that is phase weighted"
                    "reciprocal times heaviside use positive weights! Abort..."));
      AssertThrow(w_1h > 0.0,
                  ExcMessage(
                    "When using the Dirac delta function approximation that is phase weighted"
                    "reciprocal times heaviside use positive weights! Abort..."));
      AssertThrow(std::abs(w_1g - w_1h) > std::numeric_limits<number>::epsilon(),
                  ExcMessage(
                    "When using the Dirac delta function approximation that is phase weighted"
                    "reciprocal times heaviside the phase weights must differ! Abort..."));
      AssertThrow(w_2g > 0.0,
                  ExcMessage(
                    "When using the Dirac delta function approximation that is phase weighted"
                    "reciprocal times heaviside use positive weights! Abort..."));
      AssertThrow(std::abs(w_2g - w_2h) > std::numeric_limits<number>::epsilon(),
                  ExcMessage(
                    "When using the Dirac delta function approximation that is phase weighted"
                    "reciprocal times heaviside the phase weights must differ! Abort..."));
    }

    inline number
    compute_weight(const number level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

    inline VectorizedArray<number>
    compute_weight(const VectorizedArray<number> &level_set_heaviside) const override
    {
      return compute_weight_internal(level_set_heaviside);
    }

  private:
    /**
     * This function calculates the asymmetric
     *
     *           (1 - φ) w_2g + φ w_2h
     * W(φ) = --------------------------- * c_corr
     *         φ / w_1h + (1 - φ) / w_1g
     *
     */
    template <typename value_type>
    inline value_type
    compute_weight_internal(const value_type &level_set_heaviside) const
    {
      const auto w_1 = Tools::interpolate_reciprocal(level_set_heaviside, w_1g, w_1h);
      const auto w_2 = Tools::interpolate(level_set_heaviside, w_2g, w_2h);
      return w_1 * w_2 * correction_factor;
    }

    const number w_1g;
    const number w_1h;
    const number w_2g;
    const number w_2h;

    /*
     * correction factor
     *
     *           /      w_1g w_1h ln( w_1g / w_1h )
     * c_corr = | w_2g -----------------------------
     *          \              w_1g - w_1h
     *
     *                             1 / w_1g ( ln( w_1h / w_1g ) - 1 ) + 1 / w_1h  \-1
     *          + (w_2h - w_2g) * ----------------------------------------------- |
     *                                         ( 1/w_1h - 1/w_1g )²              /
     *
     */
    const number correction_factor;
  };

  template <typename number>
  std::unique_ptr<DeltaApproximationBase<number>>
  create_phase_weighted_delta_approximation(const DeltaApproximationPhaseWeightedData<number> &data)
  {
    switch (data.type)
      {
        case DiracDeltaFunctionApproximationType::norm_of_indicator_gradient:
          return nullptr;
        case DiracDeltaFunctionApproximationType::heaviside_phase_weighted:
          return std::make_unique<DeltaApproximationHeavisidePhaseWeighted<number>>(data);
        case DiracDeltaFunctionApproximationType::heavy_phase_only:
          return std::make_unique<DeltaApproximationHeavyPhaseOnly<number>>();
        case DiracDeltaFunctionApproximationType::heaviside_times_heaviside_phase_weighted:
          return std::make_unique<DeltaApproximationHeavisideTimesHeavisidePhaseWeighted<number>>(
            data);
        case DiracDeltaFunctionApproximationType::reciprocal_phase_weighted:
          return std::make_unique<DeltaApproximationReciprocalPhaseWeighted<number>>(data);
        case DiracDeltaFunctionApproximationType::reciprocal_times_heaviside_phase_weighted:
          return std::make_unique<DeltaApproximationReciprocalTimesHeavisidePhaseWeighted<number>>(
            data);
        default:
          Assert(false, ExcNotImplemented());
      }
    return nullptr;
  }
} // namespace MeltPoolDG::LevelSet
