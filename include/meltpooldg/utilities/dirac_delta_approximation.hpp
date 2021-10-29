#pragma once
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  /**
   * Asymmetric, phase weighted Dirac delta approximation function.
   *
   * This function can be used to approximate the Dirac delta function for diffuse interfaces. The
   * approximation is asymmetric and you can weigh the two phases individually. The function is
   * defined as
   *
   * δ_w(φ) = δ(φ) * W(φ).
   *
   * The function is dependent only on the heaviside representation of the level set φ.
   * The symmetric delta function δ(φ) is defined as the norm of the indicator gradient ∇φ:
   *
   *        ∇φ
   * δ = --------
   *      ||∇φ||
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
  class DeltaApproximationPhaseWeighted
  {
  public:
    DeltaApproximationPhaseWeighted(const DeltaApproximationPhaseWeightedData<number> &data)
      : w_g(data.gas_phase_weight)
      , w_h(data.heavy_phase_weight)
      , correction_factor(2. / (w_g + w_h))
    {
      AssertThrow(w_g + w_h != 0.0,
                  ExcMessage(
                    "When using a phase weighted dirac delta function approximation"
                    " type for recoil pressure, use weights that don't sum up to zero! Abort..."));
    }

    /**
     * This function returns the weight function value
     *
     *         2 ( (1 - φ) w_g + φ w_h )
     * W(φ) = ---------------------------
     *                w_g + w_h
     *
     */
    template <typename value_type>
    inline value_type
    compute_weight(const value_type &level_set_heaviside) const
    {
      return UtilityFunctions::interpolate(level_set_heaviside, w_g, w_h) * correction_factor;
    }

  private:
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
} // namespace MeltPoolDG
