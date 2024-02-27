#pragma once

#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/utilities/enum.hpp>

#include <string>

namespace MeltPoolDG::Evaporation
{
  // evaporation specific
  BETTER_ENUM(
    EvaporationModelType,
    char,
    // prescribe a (time-dependent) function for an evaporative mass flux being constant
    // over the domain
    constant,
    // calculate the evaporative mass flux from the recoil pressure
    recoil_pressure,
    // calculate the evaporative mass flux from the saturated vapor pressure
    saturated_vapor_pressure,
    // calculate the evaporative mass flux according to the model proposed by Hardt & Wondra
    hardt_wondra)

  BETTER_ENUM(EvaporationLevelSetSourceTermType,
              char,
              // calculate the interface velocity by using the liquid velocity (H(phi)=1)
              interface_velocity_sharp_heavy,
              // calculate the interface velocity by using the value at the level set 0 iso contour
              interface_velocity_sharp,
              // calculate a divergence-free interface velocity and use it to advect the level set
              interface_velocity,
              // use the source term due to evaporation as right hand-side term
              rhs)

  BETTER_ENUM(InterfaceForceType, char, diffuse, sharp)

  BETTER_ENUM(EvaporCoolingInterfaceFluxType, char, none, diffuse, sharp, sharp_conforming)

  template <typename number = double>
  struct EvaporationData
  {
    number                         evaporative_mass_flux_scale_factor = 1.0;
    std::string                    evaporative_mass_flux              = "0.0";
    number                         ls_value_liquid                    = 1.0;
    number                         ls_value_gas                       = -1.0;
    InterfaceForceType             formulation_source_term_continuity = InterfaceForceType::diffuse;
    EvaporCoolingInterfaceFluxType formulation_source_term_heat =
      EvaporCoolingInterfaceFluxType::diffuse;
    std::string formulation_evaporative_mass_flux_over_interface =
      "continuous"; // not needed if evaporation_model == "constant"
    EvaporationModelType              evaporation_model = EvaporationModelType::constant;
    number                            coefficient       = 0.0;
    unsigned int                      line_integral_n_subdivisions_per_side = 10;
    unsigned int                      line_integral_n_subdivisions_MCA      = 1;
    EvaporationLevelSetSourceTermType level_set_source_term_type =
      EvaporationLevelSetSourceTermType::interface_velocity;
    bool do_level_set_pressure_gradient_interpolation = false;

    void
    add_parameters(dealii::ParameterHandler &prm);

    void
    check_input_parameters(const unsigned int ls_n_subdivisions) const;
  };
} // namespace MeltPoolDG::Evaporation
