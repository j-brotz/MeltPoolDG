#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/evaporation/evaporation_data.hpp>

namespace MeltPoolDG::Evaporation
{

  template <typename number>
  void
  EvaporationData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("evaporation");
    {
      prm.add_parameter("evapor evaporative mass flux scale factor",
                        evaporative_mass_flux_scale_factor,
                        "Scale factor for the evaporative flux");
      prm.add_parameter(
        "evapor evaporative mass flux",
        evaporative_mass_flux,
        "For evapor evaporation model == constant, prescribe a spatially constant "
        "mass flux due to evaporation (SI unit in kg/m²s), as a function over time t "
        ", e.g. min(2.*t,0.01).");
      prm.add_parameter("evapor ls value liquid",
                        ls_value_liquid,
                        "Set the level set value corresponding to the liquid domain.",
                        dealii::Patterns::Selection("1|-1|1.|-1.|1.0|-1.0"));
      prm.add_parameter("evapor ls value gas",
                        ls_value_gas,
                        "Set the level set value corresponding to the gaseous domain.",
                        dealii::Patterns::Selection("1|-1|1.|-1.|1.0|-1.0"));
      prm.add_parameter("evapor formulation source term continuity",
                        formulation_source_term_continuity,
                        "Select how the additional source term due to evaporation in the"
                        " continuity equation is computed.");
      prm.add_parameter("evapor formulation source term heat",
                        formulation_source_term_heat,
                        "Select how the additional source term due to evaporation in the"
                        " heat equation is computed.");
      // @todo must be modified
      prm.add_parameter(
        "evapor formulation evaporative mass flux over interface",
        formulation_evaporative_mass_flux_over_interface,
        "Choose the formulation how the (local) evaporative mass flux will be converted to a DoF vector."
        "will be calculated.",
        Patterns::Selection("continuous|interface value|line integral"));
      prm.add_parameter("evapor evaporation model",
                        evaporation_model,
                        "Choose the formulation how the evaporative mass flux mDot (kg/(m2s)) "
                        "will be calculated.");
      prm.add_parameter("evapor coefficient", coefficient, "Evaporation coefficient.");
      prm.add_parameter("evapor line integral n subdivisions per side",
                        line_integral_n_subdivisions_per_side,
                        "Number of subdivisions per side to compute the points perpendicular to "
                        "the interface for the evaporative mass flux evaluation by "
                        "means of the line integral.");
      prm.add_parameter("evapor line integral n subdivisions MCA",
                        line_integral_n_subdivisions_MCA,
                        "Number of subdivisions for the marching cube algorithm within the "
                        "evaporative mass flux evaluation by means of the line integral.");
      prm.add_parameter("evapor level set source term type",
                        level_set_source_term_type,
                        "Set the type how the evaporative mass flux should be considered "
                        "in the level set equation.");
      prm.add_parameter("evapor do level set pressure gradient interpolation",
                        do_level_set_pressure_gradient_interpolation,
                        "Set if the level set gradient for computing the delta function within "
                        "the evaporative mass flux source terms should be computed based on an "
                        "interpolation to the pressure space. This is only implemented for "
                        "evapor_level_set_source_term_type = rhs.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  EvaporationData<number>::check_input_parameters(const unsigned int ls_n_subdivisions) const
  {
    /*
     *  check if level set assignment of gaseous/liquid phase is done correctly
     */
    AssertThrow(ls_value_liquid != ls_value_gas,
                ExcMessage("Parameterhandler: ls value liquid must not be equal to ls value gas."));


    AssertThrow((ls_n_subdivisions == 1 ||
                 formulation_evaporative_mass_flux_over_interface != "line integral"),
                ExcMessage(
                  "If you use the formulation of the evaporative mass flux over the interface "
                  "using the value at the interface or a line integral, n_subdivisions for the "
                  "level set must be 1."));
  }

  template struct EvaporationData<double>;
} // namespace MeltPoolDG::Evaporation