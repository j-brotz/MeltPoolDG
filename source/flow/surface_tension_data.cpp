#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/flow/surface_tension_data.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  void
  SurfaceTensionData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("surface tension");
    {
      prm.add_parameter("surface tension coefficient",
                        surface_tension_coefficient,
                        "Constant coefficient for calculating surface tension");
      prm.add_parameter("temperature dependent surface tension coefficient",
                        temperature_dependent_surface_tension_coefficient,
                        "Temperature-dependent coefficient for calculating temperetaure-dependent "
                        "surface tension (Marangoni convection)");
      prm.add_parameter("reference temperature",
                        reference_temperature,
                        "Reference temperature for calculating surface tension");
      prm.add_parameter(
        "coefficient residual fraction",
        coefficient_residual_fraction,
        "Define the minimum fraction of the constant surface tension reference value "
        "that can be reached.",
        dealii::Patterns::Double(0.0, 1.0));
      prm.add_parameter(
        "zero surface tension in solid",
        zero_surface_tension_in_solid,
        "Set this parameter to true to only apply surface tension if the solid fraction is zero.");

      delta_approximation_phase_weighted.add_parameters(prm);
      time_step_limit.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  SurfaceTensionData<number>::post(const MaterialData<number> &material)
  {
    delta_approximation_phase_weighted.set_parameters(
      material, LevelSet::ParameterScaledInterpolationType::density);
  }

  template <typename number>
  void
  SurfaceTensionData<number>::check_input_parameters(const bool curv_enable) const
  {
    // check if curvature computation is enabled in case of surface tension
    const bool do_compute_surface_tension =
      std::abs(surface_tension_coefficient) > 1e-10 ||
      std::abs(temperature_dependent_surface_tension_coefficient) > 1e-10;
    AssertThrow(!do_compute_surface_tension || curv_enable,
                dealii::ExcMessage(
                  "Curvature computation must be enabled in case of surface tension."));
  }

  template struct SurfaceTensionData<double>;
} // namespace MeltPoolDG::Flow