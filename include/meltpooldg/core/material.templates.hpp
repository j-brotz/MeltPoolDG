#pragma once

#include <meltpooldg/core/material.hpp>
//
#include <deal.II/base/exceptions.h>

#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <limits>


namespace MeltPoolDG
{
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
  inline MaterialParameterValues<value_type, number>
  Material<number>::compute_parameters(const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    Assert(
      material_type == MaterialTypes::gas or material_type == MaterialTypes::liquid,
      dealii::ExcMessage(
        "This compute_parameters() implementation should not be called for the current material type! Abort..."));

    return compute_parameters_internal(value_type(0.0), value_type(0.0), flags);
  }



  template <typename number>
  template <typename value_type>
  inline MaterialParameterValues<value_type, number>
  Material<number>::compute_parameters(const value_type                               &v,
                                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    Assert(
      material_type != MaterialTypes::gas_liquid_solid and
        material_type != MaterialTypes::gas_liquid_solid_consistent_with_evaporation,
      dealii::ExcMessage(
        "This compute_parameters() implementation should not be called for the current material type! Abort..."));

    return compute_parameters_internal(v, value_type(0.0), flags);
  }



  template <typename number>
  template <typename value_type>
  inline MaterialParameterValues<value_type, number>
  Material<number>::compute_parameters(const value_type &level_set_heaviside,
                                       const value_type &temperature,
                                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    return compute_parameters_internal(level_set_heaviside, temperature, flags);
  }



  template <typename number>
  template <typename value_type, int dim>
  inline MaterialParameterValues<value_type, number>
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
        case MaterialTypes::gas:
        case MaterialTypes::liquid:
          return compute_parameters_internal(value_type(0.0), value_type(0.0), flags);
        default:
          Assert(false, dealii::ExcNotImplemented());
          return MaterialParameterValues<dealii::VectorizedArray<number>, number>();
      }
  }



  template <typename number>
  template <typename value_type, int dim>
  inline MaterialParameterValues<value_type, number>
  Material<number>::compute_parameters(
    const FECellIntegrator<dim, 1, number>              &level_set_heaviside_val,
    const std::vector<FECellIntegrator<dim, 1, number>> &temperature_val,
    const MaterialUpdateFlags::MaterialUpdateFlags      &flags,
    const unsigned int                                   q_index) const
  {
    Assert(temperature_val.size() <= 2, dealii::ExcInternalError());

    const auto get_temperature_value = [&]() {
      if (temperature_val.size() == 1) // default case
        return temperature_val[0].get_value(q_index);
      else if (temperature_val.size() == 2) // two phase cut intersected cell case
        // For two phase intersected cut cells, temperature_val[0] contains the temperature values
        // of the liquid domain and temperature_val[1] the temperature values of the gas domain that
        // each may be ghost values. We use the level set as heaviside as in indicator to select the
        // non-ghosted values each.
        return dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than_or_equal>(
          level_set_heaviside_val.get_value(q_index),
          0.5,
          temperature_val[0].get_value(q_index),
          temperature_val[1].get_value(q_index));
      else // temperature_val.empty()
           // An empty temperature_val vector indicates that the cut case is one phase and the
           // current cell is fully in the gas domain - which doesn't have any temperature
           // information. We use the value that is just above the liquidus temperature so that
           // solid faction is reliably zero in that domain.
        return value_type(
          std::nextafter(data.liquidus_temperature, std::numeric_limits<number>::infinity()));
    };

    switch (material_type)
      {
        case MaterialTypes::gas_liquid_solid:
        case MaterialTypes::gas_liquid_solid_consistent_with_evaporation:
          return compute_parameters_internal(level_set_heaviside_val.get_value(q_index),
                                             get_temperature_value(),
                                             flags);
        case MaterialTypes::gas_liquid:
        case MaterialTypes::gas_liquid_consistent_with_evaporation:
          return compute_parameters_internal(level_set_heaviside_val.get_value(q_index),
                                             value_type(0.0),
                                             flags);
        case MaterialTypes::liquid_solid:
          return compute_parameters_internal(get_temperature_value(), value_type(0.0), flags);
        case MaterialTypes::gas:
        case MaterialTypes::liquid:
          return compute_parameters_internal(value_type(0.0), value_type(0.0), flags);
        default:
          Assert(false, dealii::ExcNotImplemented());
          return MaterialParameterValues<dealii::VectorizedArray<number>, number>();
      }
  }



  template <typename number>
  inline bool
  Material<number>::has_dependency(const FieldType &field_type) const
  {
    switch (field_type)
      {
        case FieldType::none:
          return material_type == MaterialTypes::gas or material_type == MaterialTypes::liquid;
        case FieldType::temperature:
          return material_type == MaterialTypes::liquid_solid or
                 material_type == MaterialTypes::gas_liquid_solid or
                 material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation;
        case FieldType::level_set:
          return material_type == MaterialTypes::gas_liquid or
                 material_type == MaterialTypes::gas_liquid_consistent_with_evaporation or
                 material_type == MaterialTypes::gas_liquid_solid or
                 material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation;
        default:
          AssertThrow(false, dealii::ExcNotImplemented());
          return false;
      }
  }



  template <typename number>
  template <typename value_type>
  inline MaterialParameterValues<value_type, number>
  Material<number>::compute_parameters_internal(
    const value_type                               &v1,
    const value_type                               &v2,
    const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
  {
    MaterialParameterValues<value_type, number> t;
    switch (material_type)
      {
          case MaterialTypes::gas: {
            return gas;
          }
          case MaterialTypes::liquid: {
            return liquid;
          }
        case MaterialTypes::gas_liquid:
          case MaterialTypes::gas_liquid_consistent_with_evaporation: {
            const MaterialParameterValues<value_type, number> &g = gas;
            const MaterialParameterValues<value_type, number> &l = liquid;

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
            const MaterialParameterValues<value_type, number> &l = liquid;
            const MaterialParameterValues<value_type, number> &s = solid;

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
            const MaterialParameterValues<value_type, number> &g = gas;
            const MaterialParameterValues<value_type, number> &l = liquid;
            const MaterialParameterValues<value_type, number> &s = solid;

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
    const value_type weight = data.two_phase_fluid_properties_transition_type !=
                                  TwoPhaseFluidPropertiesTransitionType::sharp ?
                                level_set_heaviside :
                                CharacteristicFunctions::heaviside(level_set_heaviside, 0.5);
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
    if (temperature_dependent_solid_fraction == value_type(0.0) or
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
    const auto weight = data.two_phase_fluid_properties_transition_type !=
                            TwoPhaseFluidPropertiesTransitionType::sharp ?
                          level_set_heaviside :
                          CharacteristicFunctions::heaviside(level_set_heaviside, 0.5);
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
  inline dealii::VectorizedArray<number>
  Material<number>::compute_temperature_dependent_solid_fraction(
    const dealii::VectorizedArray<number> &temperature) const
  {
    if (data.solid_liquid_properties_transition_type ==
        SolidLiquidPropertiesTransitionType::mushy_zone)
      return UtilityFunctions::limit_to_bounds(
        (data.liquidus_temperature - temperature) * inv_mushy_interval, 0.0, 1.0);
    else if (data.solid_liquid_properties_transition_type ==
             SolidLiquidPropertiesTransitionType::sharp)
      return dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
        temperature, data.solidus_temperature, 1.0, 0.0);
    Assert(false, dealii::ExcNotImplemented());
    return dealii::VectorizedArray<number>(0.0);
  }



  template <typename number>
  constexpr Material<number>::MaterialParameterValuesContainer::
  operator const MaterialParameterValues<number, number> &() const
  {
    return scalar_parameters;
  }



  template <typename number>
  constexpr Material<number>::MaterialParameterValuesContainer::
  operator const MaterialParameterValues<dealii::VectorizedArray<number>, number> &() const
  {
    return vectorized_parameters;
  }

} // namespace MeltPoolDG
