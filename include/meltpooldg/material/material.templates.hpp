/* ---------------------------------------------------------------------
 *
 * Author: Nils Much, Peter Munch, TUM, September 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/exceptions.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  using namespace dealii;


  /*----------------- Inline and template methods -----------------*/


  namespace MaterialUpdateFlags
  {
    inline MaterialUpdateFlags
    operator|(const MaterialUpdateFlags f1, const MaterialUpdateFlags f2)
    {
      return static_cast<MaterialUpdateFlags>(static_cast<unsigned int>(f1) |
                                              static_cast<unsigned int>(f2));
    }



    inline MaterialUpdateFlags &
    operator|=(MaterialUpdateFlags &f1, const MaterialUpdateFlags f2)
    {
      f1 = f1 | f2;
      return f1;
    }



    inline MaterialUpdateFlags
    operator&(const MaterialUpdateFlags f1, const MaterialUpdateFlags f2)
    {
      return static_cast<MaterialUpdateFlags>(static_cast<unsigned int>(f1) &
                                              static_cast<unsigned int>(f2));
    }



    inline MaterialUpdateFlags &
    operator&=(MaterialUpdateFlags &f1, const MaterialUpdateFlags f2)
    {
      f1 = f1 & f2;
      return f1;
    }

  } // namespace MaterialUpdateFlags



  template <typename number>
  template <typename value_type>
  inline MaterialParameterValues<value_type>
  Material<number>::compute_parameters(const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    Assert(
      material_type == MaterialTypes::single_phase,
      dealii::ExcMessage(
        "This compute_parameters() implementation should not be called for the current material type! Abort..."));

    return compute_parameters_internal(value_type(0.0), value_type(0.0), flags);
  }



  template <typename number>
  template <typename value_type>
  inline MaterialParameterValues<value_type>
  Material<number>::compute_parameters(const value_type                               &v,
                                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    Assert(
      material_type == MaterialTypes::gas_liquid ||
        material_type == MaterialTypes::gas_liquid_consistent_with_evaporation ||
        material_type == MaterialTypes::liquid_solid,
      dealii::ExcMessage(
        "This compute_parameters() implementation should not be called for the current material type! Abort..."));

    return compute_parameters_internal(v, value_type(0.0), flags);
  }



  template <typename number>
  template <typename value_type>
  inline MaterialParameterValues<value_type>
  Material<number>::compute_parameters(const value_type &level_set_heaviside,
                                       const value_type &temperature,
                                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    Assert(
      material_type == MaterialTypes::gas_liquid_solid ||
        material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation,
      dealii::ExcMessage(
        "This compute_parameters() implementation should not be called for the current material type! Abort..."));

    return compute_parameters_internal(level_set_heaviside, temperature, flags);
  }



  template <typename number>
  template <typename value_type, int dim>
  inline MaterialParameterValues<value_type>
  Material<number>::compute_parameters(
    const FECellIntegrator<dim, 1, number>         &level_set_heaviside_val,
    const FECellIntegrator<dim, 1, number>         &temperature_val,
    const MaterialUpdateFlags::MaterialUpdateFlags &flags,
    const unsigned int                              q_index) const
  {
    switch (material_type)
      {
        case MaterialTypes::gas_liquid_solid:
        case MaterialTypes::gas_liquid_solid_consistent_with_evaporation:
          return compute_parameters_internal(level_set_heaviside_val.get_value(q_index),
                                             temperature_val.get_value(q_index),
                                             flags);
        case MaterialTypes::gas_liquid:
        case MaterialTypes::gas_liquid_consistent_with_evaporation:
          return compute_parameters_internal(level_set_heaviside_val.get_value(q_index),
                                             value_type(0.0),
                                             flags);
        case MaterialTypes::liquid_solid:
          return compute_parameters_internal(temperature_val.get_value(q_index),
                                             value_type(0.0),
                                             flags);
        case MaterialTypes::single_phase:
          return compute_parameters_internal(value_type(0.0), value_type(0.0), flags);
        default:
          Assert(false, dealii::ExcNotImplemented());
          return MaterialParameterValues<VectorizedArray<number>>();
      }
  }



  template <typename number>
  inline bool
  Material<number>::has_dependency(const FieldType &field_type) const
  {
    switch (field_type)
      {
        case FieldType::none:
          return material_type == MaterialTypes::single_phase;
        case FieldType::temperature:
          return material_type == MaterialTypes::liquid_solid ||
                 material_type == MaterialTypes::gas_liquid_solid ||
                 material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation;
        case FieldType::level_set:
          return material_type == MaterialTypes::gas_liquid ||
                 material_type == MaterialTypes::gas_liquid_consistent_with_evaporation ||
                 material_type == MaterialTypes::gas_liquid_solid ||
                 material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation;
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
          return false;
      }
  }



  template <typename number>
  template <typename value_type>
  inline MaterialParameterValues<value_type>
  Material<number>::compute_parameters_internal(
    const value_type                               &v1,
    const value_type                               &v2,
    const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    MaterialParameterValues<value_type> t;
    switch (material_type)
      {
          case MaterialTypes::single_phase: {
            return gas;
          }
        case MaterialTypes::gas_liquid:
          case MaterialTypes::gas_liquid_consistent_with_evaporation: {
            const MaterialParameterValues<value_type> &g = gas;
            const MaterialParameterValues<value_type> &l = liquid;

            const auto &level_set_heaviside = v1;

            if (flags & MaterialUpdateFlags::thermal_conductivity)
              t.thermal_conductivity =
                compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                             g.thermal_conductivity,
                                                             l.thermal_conductivity);

            if (flags & MaterialUpdateFlags::specific_heat_capacity)
              t.specific_heat_capacity =
                compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                             g.specific_heat_capacity,
                                                             l.specific_heat_capacity);
            if (flags & MaterialUpdateFlags::density)
              {
                if (material_type == MaterialTypes::gas_liquid_consistent_with_evaporation)
                  t.density =
                    compute_two_phase_fluid_density_consistent_with_evaporation<value_type>(
                      level_set_heaviside, g.density, l.density);
                else
                  t.density = compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                                           g.density,
                                                                           l.density);
              }
            if (flags & MaterialUpdateFlags::dynamic_viscosity)
              t.dynamic_viscosity =
                compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                             g.dynamic_viscosity,
                                                             l.dynamic_viscosity);
            if (flags & MaterialUpdateFlags::volume_specific_heat_capacity)
              t.volume_specific_heat_capacity =
                compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                             g.volume_specific_heat_capacity,
                                                             l.volume_specific_heat_capacity);
            // note: For the gas-liquid phase case, the derivatives with respect to temperature
            // are zero.
            if (flags & MaterialUpdateFlags::phase_fractions)
              {
                t.gas_fraction    = value_type(1.) - level_set_heaviside;
                t.liquid_fraction = level_set_heaviside;
              }
            break;
          }
          case MaterialTypes::liquid_solid: {
            const MaterialParameterValues<value_type> &l = liquid;
            const MaterialParameterValues<value_type> &s = solid;

            const auto &temperature = v1;
            const auto  temperature_dependent_solid_fraction =
              compute_temperature_dependent_solid_fraction(temperature);

            if (flags & MaterialUpdateFlags::thermal_conductivity)
              t.thermal_conductivity = compute_solid_liquid_phases_property<value_type>(
                temperature_dependent_solid_fraction,
                l.thermal_conductivity,
                s.thermal_conductivity);
            if (flags & MaterialUpdateFlags::specific_heat_capacity)
              t.specific_heat_capacity = compute_solid_liquid_phases_property<value_type>(
                temperature_dependent_solid_fraction,
                l.specific_heat_capacity,
                s.specific_heat_capacity);
            if (flags & MaterialUpdateFlags::density)
              t.density = compute_solid_liquid_phases_property<value_type>(
                temperature_dependent_solid_fraction, l.density, s.density);
            if (flags & MaterialUpdateFlags::dynamic_viscosity)
              t.dynamic_viscosity = compute_solid_liquid_phases_property<value_type>(
                temperature_dependent_solid_fraction, l.dynamic_viscosity, s.dynamic_viscosity);
            if (flags & MaterialUpdateFlags::volume_specific_heat_capacity)
              t.volume_specific_heat_capacity = compute_solid_liquid_phases_property<value_type>(
                temperature_dependent_solid_fraction,
                l.volume_specific_heat_capacity,
                s.volume_specific_heat_capacity);
            if (flags & MaterialUpdateFlags::d_thermal_conductivity_d_T)
              t.d_thermal_conductivity_d_T =
                compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction,
                  l.thermal_conductivity,
                  s.thermal_conductivity);
            if (flags & MaterialUpdateFlags::d_specific_heat_capacity_d_T)
              t.d_specific_heat_capacity_d_T =
                compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction,
                  l.specific_heat_capacity,
                  s.specific_heat_capacity);
            if (flags & MaterialUpdateFlags::d_density_d_T)
              t.d_density_d_T =
                compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction, l.density, s.density);
            if (flags & MaterialUpdateFlags::d_volume_specific_heat_capacity_d_T)
              t.d_volume_specific_heat_capacity_d_T =
                compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction,
                  l.volume_specific_heat_capacity,
                  s.volume_specific_heat_capacity);
            if (flags & MaterialUpdateFlags::phase_fractions)
              {
                const auto hs =
                  LevelSet::Tools::interpolate_cubic(temperature_dependent_solid_fraction,
                                                     value_type(0),
                                                     value_type(1));
                t.liquid_fraction = value_type(1.) - hs;
                t.solid_fraction  = hs;
                // @note gas_fraction = 0
              }
            break;
          }
        case MaterialTypes::gas_liquid_solid:
          case MaterialTypes::gas_liquid_solid_consistent_with_evaporation: {
            const MaterialParameterValues<value_type> &g = gas;
            const MaterialParameterValues<value_type> &l = liquid;
            const MaterialParameterValues<value_type> &s = solid;

            const auto &level_set_heaviside = v1;
            const auto &temperature         = v2;
            const auto  temperature_dependent_solid_fraction =
              compute_temperature_dependent_solid_fraction(temperature);

            if (flags & MaterialUpdateFlags::thermal_conductivity)
              t.thermal_conductivity = compute_solid_liquid_gas_phases_property<value_type>(
                level_set_heaviside,
                temperature_dependent_solid_fraction,
                g.thermal_conductivity,
                l.thermal_conductivity,
                s.thermal_conductivity);
            if (flags & MaterialUpdateFlags::specific_heat_capacity)
              t.specific_heat_capacity = compute_solid_liquid_gas_phases_property<value_type>(
                level_set_heaviside,
                temperature_dependent_solid_fraction,
                g.specific_heat_capacity,
                l.specific_heat_capacity,
                s.specific_heat_capacity);
            if (flags & MaterialUpdateFlags::density)
              {
                if (material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation)
                  t.density =
                    compute_solid_liquid_gas_phases_density_consistent_with_evaporation<value_type>(
                      level_set_heaviside,
                      temperature_dependent_solid_fraction,
                      g.density,
                      l.density,
                      s.density);
                else
                  t.density = compute_solid_liquid_gas_phases_property<value_type>(
                    level_set_heaviside,
                    temperature_dependent_solid_fraction,
                    g.density,
                    l.density,
                    s.density);
              }
            if (flags & MaterialUpdateFlags::dynamic_viscosity)
              t.dynamic_viscosity = compute_solid_liquid_gas_phases_property<value_type>(
                level_set_heaviside,
                temperature_dependent_solid_fraction,
                g.dynamic_viscosity,
                l.dynamic_viscosity,
                s.dynamic_viscosity);
            if (flags & MaterialUpdateFlags::volume_specific_heat_capacity)
              t.volume_specific_heat_capacity =
                compute_solid_liquid_gas_phases_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  g.volume_specific_heat_capacity,
                  l.volume_specific_heat_capacity,
                  s.volume_specific_heat_capacity);
            if (flags & MaterialUpdateFlags::d_thermal_conductivity_d_T)
              t.d_thermal_conductivity_d_T =
                compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  l.thermal_conductivity,
                  s.thermal_conductivity);
            if (flags & MaterialUpdateFlags::d_specific_heat_capacity_d_T)
              t.d_specific_heat_capacity_d_T =
                compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  l.specific_heat_capacity,
                  s.specific_heat_capacity);
            if (flags & MaterialUpdateFlags::d_density_d_T)
              {
                if (material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation)
                  t.d_density_d_T =
                    compute_temperature_derivative_of_solid_liquid_gas_density_consistent_with_evaporation<
                      value_type>(level_set_heaviside,
                                  temperature_dependent_solid_fraction,
                                  g.density,
                                  l.density,
                                  s.density);
                else
                  t.d_density_d_T =
                    compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                      level_set_heaviside,
                      temperature_dependent_solid_fraction,
                      l.density,
                      s.density);
              }
            if (flags & MaterialUpdateFlags::d_volume_specific_heat_capacity_d_T)
              t.d_volume_specific_heat_capacity_d_T =
                compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  l.volume_specific_heat_capacity,
                  s.volume_specific_heat_capacity);
            if (flags & MaterialUpdateFlags::phase_fractions)
              {
                t.gas_fraction = value_type(1.) - level_set_heaviside;
                const auto liquid_solid_heaviside =
                  LevelSet::Tools::interpolate_cubic(temperature_dependent_solid_fraction,
                                                     value_type(0),
                                                     value_type(1));
                t.liquid_fraction = (1. - liquid_solid_heaviside) * level_set_heaviside;
                t.solid_fraction  = liquid_solid_heaviside * level_set_heaviside;
              }
            break;
          }
          default: {
            Assert(false, dealii::ExcNotImplemented());
          }
      }

    return t;
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::compute_two_phase_fluid_property(const value_type &level_set_heaviside,
                                                     const value_type &gas_value,
                                                     const value_type &liquid_solid_value) const
  {
    const value_type weight = (data.two_phase_fluid_properties_transition_type !=
                               TwoPhaseFluidPropertiesTransitionType::sharp) ?
                                level_set_heaviside :
                                UtilityFunctions::heaviside(level_set_heaviside, 0.5);
    return LevelSet::Tools::interpolate(weight, gas_value, liquid_solid_value);
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::compute_two_phase_fluid_density_consistent_with_evaporation(
    const value_type &level_set_heaviside,
    const value_type &gas_density,
    const value_type &liquid_solid_density) const
  {
    return LevelSet::Tools::interpolate_reciprocal(level_set_heaviside,
                                                   gas_density,
                                                   liquid_solid_density);
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::compute_solid_liquid_phases_property(
    const value_type &temperature_dependent_solid_fraction,
    const value_type &liquid_value,
    const value_type &solid_value) const
  {
    if (temperature_dependent_solid_fraction == value_type(0.0))
      return liquid_value;
    if (temperature_dependent_solid_fraction == value_type(1.0))
      return solid_value;
    return LevelSet::Tools::interpolate_cubic(temperature_dependent_solid_fraction,
                                              liquid_value,
                                              solid_value);
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::compute_temperature_derivative_of_solid_liquid_phases_property(
    const value_type &temperature_dependent_solid_fraction,
    const value_type &liquid_value,
    const value_type &solid_value) const
  {
    if (temperature_dependent_solid_fraction == value_type(0.0) ||
        temperature_dependent_solid_fraction == value_type(1.0))
      return value_type(0.0);
    return -1.0 * inv_mushy_interval *
           LevelSet::Tools::interpolate_cubic_derivative(temperature_dependent_solid_fraction,
                                                         liquid_value,
                                                         solid_value);
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::compute_solid_liquid_gas_phases_property(
    const value_type &level_set_heaviside,
    const value_type &temperature_dependent_solid_fraction,
    const value_type &gas_value,
    const value_type &liquid_value,
    const value_type &solid_value) const
  {
    const auto liquid_solid_value =
      compute_solid_liquid_phases_property(temperature_dependent_solid_fraction,
                                           liquid_value,
                                           solid_value);
    return compute_two_phase_fluid_property(level_set_heaviside, gas_value, liquid_solid_value);
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::compute_solid_liquid_gas_phases_density_consistent_with_evaporation(
    const value_type &level_set_heaviside,
    const value_type &temperature_dependent_solid_fraction,
    const value_type &gas_value,
    const value_type &liquid_value,
    const value_type &solid_value) const
  {
    const auto liquid_solid_value =
      compute_solid_liquid_phases_property(temperature_dependent_solid_fraction,
                                           liquid_value,
                                           solid_value);
    return compute_two_phase_fluid_density_consistent_with_evaporation(level_set_heaviside,
                                                                       gas_value,
                                                                       liquid_solid_value);
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::compute_temperature_derivative_of_solid_liquid_gas_property(
    const value_type &level_set_heaviside,
    const value_type &temperature_dependent_solid_fraction,
    const value_type &liquid_value,
    const value_type &solid_value) const
  {
    const auto temp = compute_temperature_derivative_of_solid_liquid_phases_property(
      temperature_dependent_solid_fraction, liquid_value, solid_value);
    const auto weight = (data.two_phase_fluid_properties_transition_type !=
                         TwoPhaseFluidPropertiesTransitionType::sharp) ?
                          level_set_heaviside :
                          UtilityFunctions::heaviside(level_set_heaviside, 0.5);
    return temp * weight;
  }



  template <typename number>
  template <typename value_type>
  inline value_type
  Material<number>::
    compute_temperature_derivative_of_solid_liquid_gas_density_consistent_with_evaporation(
      const value_type &level_set_heaviside,
      const value_type &temperature_dependent_solid_fraction,
      const value_type &gas_value,
      const value_type &liquid_value,
      const value_type &solid_value) const
  {
    const auto liquid_solid_value =
      compute_solid_liquid_phases_property(temperature_dependent_solid_fraction,
                                           liquid_value,
                                           solid_value);
    const auto liquid_solid_derivative =
      compute_temperature_derivative_of_solid_liquid_phases_property(
        temperature_dependent_solid_fraction, liquid_value, solid_value);
    const auto temp =
      liquid_solid_value * (1. + level_set_heaviside * (gas_value / liquid_solid_value - 1.));
    return level_set_heaviside * gas_value * gas_value * liquid_solid_derivative / (temp * temp);
  }



  template <typename number>
  inline number
  Material<number>::compute_temperature_dependent_solid_fraction(const number temperature) const
  {
    if (data.solid_liquid_properties_transition_type ==
        SolidLiquidPropertiesTransitionType::mushy_zone)
      return UtilityFunctions::limit_to_bounds(
        (data.liquidus_temperature - temperature) * inv_mushy_interval, 0.0, 1.0);
    else if (data.solid_liquid_properties_transition_type ==
             SolidLiquidPropertiesTransitionType::sharp)
      return temperature < data.solidus_temperature ? 1.0 : 0.0;
    Assert(false, dealii::ExcNotImplemented());
    return 0.0;
  }



  template <typename number>
  inline VectorizedArray<number>
  Material<number>::compute_temperature_dependent_solid_fraction(
    const VectorizedArray<number> &temperature) const
  {
    if (data.solid_liquid_properties_transition_type ==
        SolidLiquidPropertiesTransitionType::mushy_zone)
      return UtilityFunctions::limit_to_bounds(
        (data.liquidus_temperature - temperature) * inv_mushy_interval, 0.0, 1.0);
    else if (data.solid_liquid_properties_transition_type ==
             SolidLiquidPropertiesTransitionType::sharp)
      return compare_and_apply_mask<SIMDComparison::less_than>(temperature,
                                                               data.solidus_temperature,
                                                               1.0,
                                                               0.0);
    Assert(false, dealii::ExcNotImplemented());
    return VectorizedArray<number>(0.0);
  }



  template <typename number>
  constexpr Material<number>::MaterialParameterValuesContainer::
  operator const MaterialParameterValues<number> &() const
  {
    return scalar_parameters;
  }



  template <typename number>
  constexpr Material<number>::MaterialParameterValuesContainer::
  operator const MaterialParameterValues<VectorizedArray<number>> &() const
  {
    return vectorized_parameters;
  }

} // namespace MeltPoolDG
