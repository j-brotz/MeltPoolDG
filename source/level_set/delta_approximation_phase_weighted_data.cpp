#include <deal.II/base/exceptions.h>

#include <meltpooldg/level_set/delta_approximation_phase_weighted_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  void
  DeltaApproximationPhaseWeightedData<number>::add_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("dirac delta function approximation");
    {
      prm.add_parameter("type", type, "Choose how to smear a parameter over the interface.");
      prm.add_parameter("auto weights",
                        auto_weights,
                        "Choose if weights should be computed automatically.");
      prm.add_parameter(
        "gas phase weight",
        gas_phase_weight,
        "If >>> dirac delta function approximation type <<< is set to any phase weighted option"
        "this parameter controls the (first) weight of the gas phase (level set = -1).");
      prm.add_parameter(
        "heavy phase weight",
        heavy_phase_weight,
        "If >>> dirac delta function approximation type <<< is set to any phase weighted option"
        "this parameter controls the (first) weight of the heavy phase (level set = 1).");
      prm.add_parameter(
        "gas phase weight 2",
        gas_phase_weight_2,
        "If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< "
        "this parameter controls the second weight of the gas phase (level set = -1).");
      prm.add_parameter(
        "heavy phase weight 2",
        heavy_phase_weight_2,
        "If >>> dirac delta function approximation type <<< is set to >>> heaviside_times_heaviside_phase_weighted <<< "
        "this parameter controls the second weight of the heavy liquid/solid phase (level set = 1).");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  DeltaApproximationPhaseWeightedData<number>::set_parameters(
    const MaterialData<number>             &material,
    const ParameterScaledInterpolationType &interpolation_type)
  {
    if (!auto_weights)
      return;

    switch (interpolation_type)
      {
          case ParameterScaledInterpolationType::volume_specific_heat_capacity: {
            gas_phase_weight   = material.gas.density * material.gas.specific_heat_capacity;
            heavy_phase_weight = material.liquid.density * material.liquid.specific_heat_capacity;
            break;
          }
          case ParameterScaledInterpolationType::specific_heat_capacity_times_density: {
            gas_phase_weight     = material.gas.density;
            heavy_phase_weight   = material.liquid.density;
            gas_phase_weight_2   = material.gas.specific_heat_capacity;
            heavy_phase_weight_2 = material.liquid.specific_heat_capacity;
            break;
          }
          case ParameterScaledInterpolationType::density: {
            gas_phase_weight   = material.gas.density;
            heavy_phase_weight = material.liquid.density;
            break;
          }
        default:
          AssertThrow(false, ExcNotImplemented());
      }
  }

  template struct DeltaApproximationPhaseWeightedData<double>;
} // namespace MeltPoolDG::LevelSet