#pragma once
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

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

    template <typename value_type>
    inline value_type
    compute_weight(const value_type &level_set_heaviside) const
    {
      return UtilityFunctions::interpolate(level_set_heaviside, w_g, w_h) * correction_factor;
    }

  private:
    const number w_g;
    const number w_h;
    const number correction_factor;
  };
} // namespace MeltPoolDG
