/* ---------------------------------------------------------------------
 *
 * Author: Nils Much, Peter Munch, TUM, September 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/base/vectorization.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <typename value_type>
  struct MaterialParameterValues
  {
    value_type capacity{0};
    value_type conductivity{0};
    value_type density{0};
    value_type viscosity{0};
    value_type d_capacity_d_T{0};
    value_type d_conductivity_d_T{0};
    value_type d_density_d_T{0};
    value_type gas_fraction{0};
    value_type liquid_fraction{0};
    value_type solid_fraction{0};
  };

  enum class MaterialTypes
  {
    single_phase,
    gas_liquid,
    gas_liquid_consistent_with_evaporation,
    liquid_solid,
    gas_liquid_solid,
    gas_liquid_solid_consistent_with_evaporation
  };

  namespace MaterialUpdateFlags
  {
    enum MaterialUpdateFlags
    {
      none               = 0,
      capacity           = 1 << 0,
      conductivity       = 1 << 1,
      density            = 2 << 2,
      viscosity          = 2 << 3,
      d_capacity_d_T     = 2 << 4,
      d_conductivity_d_T = 2 << 5,
      d_density_d_T      = 2 << 6,
      phase_fractions    = 2 << 7
    };

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
  class Material
  {
  public:
    Material(const MaterialData<number> &material_data, const MaterialTypes material_type)
      : data(material_data)
      , material_type(material_type) // TODO determine material type depending on MaterialData
    {}

    /**
     * This overload of compute_parameters() can be used only in the context of single phase
     * simulations.
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type>
    compute_parameters(const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
    {
      Assert(
        material_type == MaterialTypes::single_phase,
        ExcMessage(
          "This compute_parameters() implementation should not be called for the current material type! Abort..."));

      return compute_parameters_internal(value_type(0.0), value_type(0.0), flags);
    }

    /**
     * This overload of get_parameters() can be used only in the context of two-phase simulations
     * (gas-liquid or liquid-solid).
     *
     * If the material_type is MaterialTypes::gas_liquid, @p v is the heaviside representation of the
     * level set, if the material_type is MaterialTypes::liquid_solid, @p v is the temperature.
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type>
    compute_parameters(const value_type &                              v,
                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
    {
      Assert(
        material_type == MaterialTypes::gas_liquid ||
          material_type == MaterialTypes::gas_liquid_consistent_with_evaporation ||
          material_type == MaterialTypes::liquid_solid,
        ExcMessage(
          "This compute_parameters() implementation should not be called for the current material type! Abort..."));

      return compute_parameters_internal(v, value_type(0.0), flags);
    }

    /**
     * This overload of get_parameters() can be used only in the context of three-phase simulations
     * (gas-liquid-solid).
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type>
    compute_parameters(const value_type &                              level_set_heaviside,
                       const value_type &                              temperature,
                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
    {
      Assert(
        material_type == MaterialTypes::gas_liquid_solid ||
          material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation,
        ExcMessage(
          "This compute_parameters() implementation should not be called for the current material type! Abort..."));

      return compute_parameters_internal(level_set_heaviside, temperature, flags);
    }

  private:
    /**
     * Get arbitrary parameters, specified by @p flags for the currently active material_type.
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type>
    compute_parameters_internal(const value_type &                              v1,
                                const value_type &                              v2,
                                const MaterialUpdateFlags::MaterialUpdateFlags &flags) const
    {
      MaterialParameterValues<value_type> t;
      switch (material_type)
        {
            case MaterialTypes::single_phase: {
              if (flags & MaterialUpdateFlags::capacity)
                t.capacity = data.first.capacity;
              if (flags & MaterialUpdateFlags::conductivity)
                t.conductivity = data.first.conductivity;
              if (flags & MaterialUpdateFlags::density)
                t.density = data.first.density;
              if (flags & MaterialUpdateFlags::viscosity)
                t.viscosity = data.first.viscosity;
              // note: For the single phase case, the derivatives with respect to temperature are
              // zero.
              if (flags & MaterialUpdateFlags::phase_fractions)
                t.gas_fraction = 1.0;
              break;
            }
          case MaterialTypes::gas_liquid:
            case MaterialTypes::gas_liquid_consistent_with_evaporation: {
              const auto &level_set_heaviside = v1;

              if (flags & MaterialUpdateFlags::capacity)
                t.capacity = compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                                          data.first.capacity,
                                                                          data.second.capacity);
              if (flags & MaterialUpdateFlags::conductivity)
                t.conductivity =
                  compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                               data.first.conductivity,
                                                               data.second.conductivity);
              if (flags & MaterialUpdateFlags::density)
                {
                  if (material_type == MaterialTypes::gas_liquid_consistent_with_evaporation)
                    t.density =
                      compute_two_phase_fluid_density_consistent_with_evaporation<value_type>(
                        level_set_heaviside, data.first.density, data.second.density);
                  else
                    t.density = compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                                             data.first.density,
                                                                             data.second.density);
                }
              if (flags & MaterialUpdateFlags::viscosity)
                t.viscosity = compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                                           data.first.viscosity,
                                                                           data.second.viscosity);
              // note: For the gas-liquid phase case, the derivatives with respect to temperature
              // are zero.
              if (flags & MaterialUpdateFlags::phase_fractions)
                {
                  t.gas_fraction    = 1. - level_set_heaviside;
                  t.liquid_fraction = level_set_heaviside;
                }
              break;
            }
            case MaterialTypes::liquid_solid: {
              const auto &temperature = v1;
              const auto  temperature_dependent_solid_fraction =
                compute_temperature_dependent_solid_fraction(temperature);

              if (flags & MaterialUpdateFlags::capacity)
                t.capacity = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction, data.second.capacity, data.solid.capacity);
              if (flags & MaterialUpdateFlags::conductivity)
                t.conductivity = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction,
                  data.second.conductivity,
                  data.solid.conductivity);
              if (flags & MaterialUpdateFlags::density)
                t.density = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction, data.second.density, data.solid.density);
              if (flags & MaterialUpdateFlags::viscosity)
                t.viscosity = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction,
                  data.second.viscosity,
                  data.solid.viscosity);
              if (flags & MaterialUpdateFlags::d_capacity_d_T)
                t.d_capacity_d_T =
                  compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                    temperature_dependent_solid_fraction,
                    data.second.capacity,
                    data.solid.capacity);
              if (flags & MaterialUpdateFlags::d_conductivity_d_T)
                t.d_conductivity_d_T =
                  compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                    temperature_dependent_solid_fraction,
                    data.second.conductivity,
                    data.solid.conductivity);
              if (flags & MaterialUpdateFlags::d_density_d_T)
                t.d_density_d_T =
                  compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                    temperature_dependent_solid_fraction, data.second.density, data.solid.density);
              if (flags & MaterialUpdateFlags::phase_fractions)
                {
                  t.liquid_fraction = 1. - temperature_dependent_solid_fraction;
                  t.solid_fraction  = temperature_dependent_solid_fraction;
                  // @note: gas_fraction = 0
                }
              break;
            }
          case MaterialTypes::gas_liquid_solid:
            case MaterialTypes::gas_liquid_solid_consistent_with_evaporation: {
              const auto &level_set_heaviside = v1;
              const auto &temperature         = v2;
              const auto  temperature_dependent_solid_fraction =
                compute_temperature_dependent_solid_fraction(temperature);

              if (flags & MaterialUpdateFlags::capacity)
                t.capacity = compute_solid_liquid_gas_phases_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  data.first.capacity,
                  data.second.capacity,
                  data.solid.capacity);
              if (flags & MaterialUpdateFlags::conductivity)
                t.conductivity = compute_solid_liquid_gas_phases_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  data.first.conductivity,
                  data.second.conductivity,
                  data.solid.conductivity);
              if (flags & MaterialUpdateFlags::density)
                {
                  if (material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation)
                    t.density = compute_solid_liquid_gas_phases_density_consistent_with_evaporation<
                      value_type>(level_set_heaviside,
                                  temperature_dependent_solid_fraction,
                                  data.first.density,
                                  data.second.density,
                                  data.solid.density);
                  else
                    t.density = compute_solid_liquid_gas_phases_property<value_type>(
                      level_set_heaviside,
                      temperature_dependent_solid_fraction,
                      data.first.density,
                      data.second.density,
                      data.solid.density);
                }
              if (flags & MaterialUpdateFlags::viscosity)
                t.viscosity = compute_solid_liquid_gas_phases_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  data.first.viscosity,
                  data.second.viscosity,
                  data.solid.viscosity);
              if (flags & MaterialUpdateFlags::d_capacity_d_T)
                t.d_capacity_d_T =
                  compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                    level_set_heaviside,
                    temperature_dependent_solid_fraction,
                    data.second.capacity,
                    data.solid.capacity);
              if (flags & MaterialUpdateFlags::d_conductivity_d_T)
                t.d_conductivity_d_T =
                  compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                    level_set_heaviside,
                    temperature_dependent_solid_fraction,
                    data.second.conductivity,
                    data.solid.conductivity);
              if (flags & MaterialUpdateFlags::d_density_d_T)
                {
                  if (material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation)
                    t.d_density_d_T =
                      compute_temperature_derivative_of_solid_liquid_gas_density_consistent_with_evaporation<
                        value_type>(level_set_heaviside,
                                    temperature_dependent_solid_fraction,
                                    data.first.density,
                                    data.second.density,
                                    data.solid.density);
                  else
                    t.d_density_d_T =
                      compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                        level_set_heaviside,
                        temperature_dependent_solid_fraction,
                        data.second.density,
                        data.solid.density);
                }
              if (flags & MaterialUpdateFlags::phase_fractions)
                {
                  t.gas_fraction = 1. - level_set_heaviside;
                  t.liquid_fraction =
                    (1. - temperature_dependent_solid_fraction) * level_set_heaviside;
                  t.solid_fraction = temperature_dependent_solid_fraction * level_set_heaviside;
                }
              break;
            }
            default: {
              Assert(false, ExcNotImplemented());
            }
        }

      return t;
    }

    /**
     * Determine a material parameter for two phase flow. If level_set_heaviside = 0 this
     * function sets the parameter equal to the @p gas_value and if @p ls_heaviside_val = 1 it
     * sets the parameter equal to the @p liquid_solid_value. At the interface, the
     * parameter jumps if "material two phase properties transition type" is set to
     * "sharp", otherwise the parameter is distributed according to the level set function.
     *
     * The distributed parameter x of the two-phase fluid is then computed by
     *
     * x = (1-ls) * x_g + ls * x_l
     *
     * with the heaviside representation of the level set function ls (level_set_heaviside),
     * the value of the gaseous phase x_g (@p gas_value) and the value of the liquid (and
     * solid) phase x_l (@p liquid_solid_value).
     *
     * @note This does not account for solidification/melting effects. In case of
     * solidification/melting the value of @p liquid_solid_value must be set to the liquid/solid
     * phase's (level set = 1) value, as determined by
     * compute_solid_liquid_phases_property().
     */
    template <typename value_type>
    inline value_type
    compute_two_phase_fluid_property(const value_type &level_set_heaviside,
                                     const value_type &gas_value,
                                     const value_type &liquid_solid_value) const
    {
      const value_type weight = (data.two_phase_properties_transition_type !=
                                 TwoPhaseFluidPropertiesTransitionType::sharp) ?
                                  level_set_heaviside :
                                  UtilityFunctions::heaviside(level_set_heaviside, 0.5);
      return UtilityFunctions::interpolate(weight, gas_value, liquid_solid_value);
    }

    /**
     * Determine the density for two phase flow consistent with mass flux due to evaporation.  If
     * level_set_heaviside = 0 this function sets the parameter equal to the @p gas_density and if
     * @p ls_heaviside_val = 1 it sets the parameter equal to the @p liquid_solid_density. At the
     * interface, the density is distributed following a reciprocal distribution
     *
     * 1       ls        (1-ls)
     * --- =  ---   + --------
     *  x      x_l        x_g
     *
     * with the heaviside representation of the level set function ls (level_set_heaviside),
     * the density of the gaseous phase x_g (@p gas_density) and the density of the liquid (and
     * solid) phase x_l (@p liquid_solid_density).
     *
     * @note This does not account for solidification/melting effects. In case of
     * solidification/melting the value of @p liquid_solid_density must be set to the mixed liquid/solid
     * phase's (level set = 1) density, as determined by
     * compute_solid_liquid_phases_property().
     */
    template <typename value_type>
    inline value_type
    compute_two_phase_fluid_density_consistent_with_evaporation(
      const value_type &level_set_heaviside,
      const value_type &gas_density,
      const value_type &liquid_solid_density) const
    {
      return gas_density / (1. + (gas_density / liquid_solid_density - 1.) * level_set_heaviside);
    }

    /**
     * Determine a material parameter for the solid/liquid phases. In the mushy zone (where
     * the solid_fraction is between 0 and 1) the material parameter will be interpolated
     * with smooth cubic function, see UtilityFunctions::interpolate_cubic():
     *
     * The parameter x of the solid/liquid phases is computed by the cubic spline
     * interpolation:
     *
     * x = x_l + (x_s - x_l) * (-2*sf^3 + 3*sf^2)
     *
     * with the solid_fraction sf, the value of the solid phase x_s (@p solid_value) and the
     * value of the liquid phase x_l (@p liquid_value).
     *
     * @note This function does not consider two-phase flow material parameters. However, the result
     * of this function can be used as the liquid/solid phases' (level set = 1) material
     * properties in compute_two_phase_fluid_property().
     */
    template <typename value_type>
    inline value_type
    compute_solid_liquid_phases_property(const value_type &temperature_dependent_solid_fraction,
                                         const value_type &liquid_value,
                                         const value_type &solid_value) const
    {
      if (temperature_dependent_solid_fraction == value_type(0.0))
        return liquid_value;
      if (temperature_dependent_solid_fraction == value_type(1.0))
        return solid_value;
      return UtilityFunctions::interpolate_cubic(temperature_dependent_solid_fraction,
                                                 liquid_value,
                                                 solid_value);
    }

    /**
     * Determine the derivative of a material parameter for the solid/liquid phases, calculated by
     * compute_solid_liquid_phases_property(), with respect to the temperature, given as
     *
     *  dx                                       d sf
     * ----  = (x_s - x_l) * (-6*sf^2 + 6*sf) * ------
     *  dT                                        dT
     *
     *  d sf          -1
     * ------ = --------------- , if T_sol < T < T_liq , 0 otherwise
     *   dT      T_liq - T_sol
     *
     * with the solid_fraction sf, the value of the solid phase x_s (@p solid_value), the
     * value of the liquid phase x_l (@p liquid_value), the liquidus temperature T_liq and
     * the solidus temperature T_sol.
     *
     * @note This function does not consider two-phase flow material parameters. However, the result
     * of this function can be used as the liquid/solid phases' (level set = 1) material
     * properties.
     */
    template <typename value_type>
    inline value_type
    compute_temperature_derivative_of_solid_liquid_phases_property(
      const value_type &temperature_dependent_solid_fraction,
      const value_type &liquid_value,
      const value_type &solid_value) const
    {
      if (temperature_dependent_solid_fraction == value_type(0.0) ||
          temperature_dependent_solid_fraction == value_type(1.0))
        return value_type(0.0);
      return -1.0 * data.inv_mushy_interval *
             UtilityFunctions::interpolate_cubic_derivative(temperature_dependent_solid_fraction,
                                                            liquid_value,
                                                            solid_value);
    }

    /**
     * Determine a material parameter with respect to two-phase flow and solidification
     * effects.
     *
     * See compute_solid_liquid_phases_property() and compute_two_phase_fluid_property()
     * for more detail.
     *
     * parameter Parameter to be determined.
     * @p gas_value The parameter's value in the gaseous phase
     * @p liquid_value The parameter's value in the liquid phase
     * @p solid_value The parameter's value in the solid phase
     */
    template <typename value_type>
    inline value_type
    compute_solid_liquid_gas_phases_property(const value_type &level_set_heaviside,
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

    /**
     * Determine the density of a material exhibiting solid/liquid/gas phases consistent with mass
     * flux due to evaporation effects consistent with evaporation.
     *
     * See compute_solid_liquid_phases_property() and
     * compute_two_phase_fluid_density_consistent_with_evaporation() for more detail.
     *
     * parameter Parameter to be determined.
     * @p gas_value The parameter's value in the gaseous phase
     * @p liquid_value The parameter's value in the liquid phase
     * @p solid_value The parameter's value in the solid phase
     */
    template <typename value_type>
    inline value_type
    compute_solid_liquid_gas_phases_density_consistent_with_evaporation(
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

    /**
     * Determine the derivative of a material parameter with respect to two phase flow and
     * solidification/melting. This function will return the temperature derivatives of
     * the values determined by compute_solid_liquid_gas_phases_property(). At
     * the
     * interface, the derivative jumps to zero if "material two phase properties transition
     * type" is set to "sharp", otherwise the derivative is smeared consistently.
     *
     * The smeared derivative dx_d_T is then computed by
     *
     *  dx          dx_ls
     * ---- = ls * -------
     *  dT           dT
     *
     * with the heaviside representation of the level set function ls (level_set_heaviside),
     * the derivative of the liquid(and solid) phase x_ls, which is determined by
     * compute_temperature_derivative_of_solid_liquid_phases_property().
     *
     * @note The value in the gaseous phase are constant with respect to the temperature, thus its
     * derivative can be neglected.
     *
     * derivative Derivative to be determined.
     * @p liquid_value The parameter's value in the liquid phase
     * @p solid_value The parameter's value in the solid phase
     */
    template <typename value_type>
    inline value_type
    compute_temperature_derivative_of_solid_liquid_gas_property(
      const value_type &level_set_heaviside,
      const value_type &temperature_dependent_solid_fraction,
      const value_type &liquid_value,
      const value_type &solid_value) const
    {
      const auto temp = compute_temperature_derivative_of_solid_liquid_phases_property(
        temperature_dependent_solid_fraction, liquid_value, solid_value);
      const auto weight = (data.two_phase_properties_transition_type !=
                           TwoPhaseFluidPropertiesTransitionType::sharp) ?
                            level_set_heaviside :
                            UtilityFunctions::heaviside(level_set_heaviside, 0.5);
      return temp * weight;
    }

    /**
     * Determine the derivative of a material parameter with respect to the temperature for
     * two phase flow and solidification/melting  consistent with the evaporation. This
     * function will return the temperature derivatives of the values determined by
     * compute_solid_liquid_gas_phases_density_consistent_with_evaporation(). At
     * the
     * interface, the derivative is smeared consistently.
     *
     * The derivative dx_d_T of the solid/liquid mixture is then computed by
     *
     *  dx                      ls * x_g²                      d x_ls
     * ---- = --------------------------------------------- * --------
     *  dT     ( x_ls * ( 1 + ls * ( x_g / x_ls - 1 ) ) )²       dT
     *
     * with the heaviside representation of the level set function ls (level_set_heaviside),
     * the value of the gaseous phase x_g (@p gas_value), the value of the liquid(and solid)
     * phase x_ls as determined by compute_solid_liquid_phases_property() and its
     * temperature derivative dx_ls_d_T as determined by
     * compute_temperature_derivative_of_solid_liquid_phases_property().
     *
     * derivative Derivative to be determined.
     * @p gas_value The parameter's value in the gaseous phase
     * @p liquid_value The parameter's value in the liquid phase
     * @p solid_value The parameter's value in the solid phase
     */
    template <typename value_type>
    inline value_type
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

    /**
     * Compute the solid fraction depending on the solidification type.
     *
     * solidification type = mushy zone
     *   If the temperature is greater than the liquidus temperature, then the solid fraction is
     *   zero. If the temperature is less than the solidus temperature, then the solid fraction is
     *   one. In between there is a linear interpolation.
     *
     * solidification type = melting point
     *   The solid fraction is zero above the melting point and one below the melting point.
     *
     * @note The solid fraction this function computes is not informed by the level-set. In the case
     *       of solidification/melting and two-phase flow, the gaseous phase is ignored.
     */
    inline number
    compute_temperature_dependent_solid_fraction(const number temperature) const
    {
      if (data.solidification_type == SolidLiquidPropertiesTransitionType::mushy_zone)
        return UtilityFunctions::limit_to_bounds(
          (data.liquidus_temperature - temperature) * data.inv_mushy_interval, 0.0, 1.0);
      else if (data.solidification_type == SolidLiquidPropertiesTransitionType::sharp)
        return temperature < data.melting_point ? 1.0 : 0.0;
      AssertThrow(false, ExcNotImplemented());
    }

    inline VectorizedArray<number>
    compute_temperature_dependent_solid_fraction(const VectorizedArray<number> &temperature) const
    {
      if (data.solidification_type == SolidLiquidPropertiesTransitionType::mushy_zone)
        return UtilityFunctions::limit_to_bounds(
          (data.liquidus_temperature - temperature) * data.inv_mushy_interval, 0.0, 1.0);
      else if (data.solidification_type == SolidLiquidPropertiesTransitionType::sharp)
        return compare_and_apply_mask<SIMDComparison::less_than>(temperature,
                                                                 data.melting_point,
                                                                 1.0,
                                                                 0.0);
      AssertThrow(false, ExcNotImplemented());
    }

    const MaterialData<number> &data;

    const MaterialTypes material_type;
  };
} // namespace MeltPoolDG
