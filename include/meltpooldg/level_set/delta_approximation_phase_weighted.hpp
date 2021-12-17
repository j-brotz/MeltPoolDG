#pragma once
#include <deal.II/base/vectorization.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_parameters.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
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
      return UtilityFunctions::interpolate(level_set_heaviside, w_g, w_h) * correction_factor;
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
   * Asymmetric, quadratic phase weighted Dirac delta approximation function.
   *
   * This function can be used to approximate the Dirac delta function for diffuse interfaces. The
   * approximation is asymmetric and you can weigh the two phases individually. The function is
   * defined as
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
   *         3 ( (1 - φ) w_g + φ w_h )²
   * W(φ) = ---------------------------- ,
   *            w_g² + w_g w_h + w_h²
   *
   * where w_g is the weight of the gaseous phase (at level set = -1) and w_h is the weight of the
   * heavy phase (at level set = 1). The weights can be chosen arbitrarily, as long as
   * w_g² + w_g w_h + w_h² != 0.
   *
   */
  template <typename number>
  class DeltaApproximationQuadHeavisidePhaseWeighted : public DeltaApproximationBase<number>
  {
  public:
    DeltaApproximationQuadHeavisidePhaseWeighted(
      const DeltaApproximationPhaseWeightedData<number> &data)
      : w_g(data.gas_phase_weight)
      , w_h(data.heavy_phase_weight)
      , correction_factor(3. / (w_g * w_g + w_g * w_h + w_h * w_h))
    {
      AssertThrow(std::abs(w_g * w_g + w_g * w_h + w_h * w_h) >
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
     *         3 ( (1 - φ) w_g + φ w_h )²
     * W(φ) = ----------------------------
     *            w_g² + w_g w_h + w_h²
     *
     */
    template <typename value_type>
    inline value_type
    compute_weight_internal(const value_type &level_set_heaviside) const
    {
      const auto temp = UtilityFunctions::interpolate(level_set_heaviside, w_g, w_h);
      return temp * temp * correction_factor;
    }

    const number w_g;
    const number w_h;

    /*
     * correction factor
     *
     *            3
     * -----------------------
     *  w_g² + w_g w_h + w_h²
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
      const auto w_1 = UtilityFunctions::interpolate(level_set_heaviside, w_1g, w_1h);
      const auto w_2 = UtilityFunctions::interpolate(level_set_heaviside, w_2g, w_2h);
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
      // clang-format off
      return correction_factor
             / // ----------------------------------------------------------
             (level_set_heaviside / w_h + (1. - level_set_heaviside) / w_g);
      // clang-format on
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
        case DiracDeltaFunctionApproximationType::quad_heaviside_phase_weighted:
          return std::make_unique<DeltaApproximationQuadHeavisidePhaseWeighted<number>>(data);
        case DiracDeltaFunctionApproximationType::heaviside_times_heaviside_phase_weighted:
          return std::make_unique<DeltaApproximationHeavisideTimesHeavisidePhaseWeighted<number>>(
            data);
        case DiracDeltaFunctionApproximationType::reciprocal_phase_weighted:
          return std::make_unique<DeltaApproximationReciprocalPhaseWeighted<number>>(data);
        default:
          Assert(false, ExcNotImplemented());
      }
    return nullptr;
  }
} // namespace MeltPoolDG
