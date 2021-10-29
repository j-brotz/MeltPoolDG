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
    MaterialParameterValues(const double capacity           = 0.0,
                            const double conductivity       = 0.0,
                            const double density            = 0.0,
                            const double viscosity          = 0.0,
                            const double d_capacity_d_T     = 0.0,
                            const double d_conductivity_d_T = 0.0,
                            const double d_density_d_T      = 0.0,
                            const double gas_fraction       = 0.0,
                            const double liquid_fraction    = 0.0,
                            const double solid_fraction     = 0.0)
      : capacity(capacity)
      , conductivity(conductivity)
      , density(density)
      , viscosity(viscosity)
      , d_capacity_d_T(d_capacity_d_T)
      , d_conductivity_d_T(d_conductivity_d_T)
      , d_density_d_T(d_density_d_T)
      , gas_fraction(gas_fraction)
      , liquid_fraction(liquid_fraction)
      , solid_fraction(solid_fraction)
    {}

    template <typename material_phase_data_struct>
    MaterialParameterValues(const material_phase_data_struct &data)
      : capacity(data.capacity)
      , conductivity(data.conductivity)
      , density(data.density)
      , viscosity(data.viscosity)
      , d_capacity_d_T(0.0)
      , d_conductivity_d_T(0.0)
      , d_density_d_T(0.0)
      , gas_fraction(1.0)
      , liquid_fraction(0.0)
      , solid_fraction(0.0)
    {}

    value_type capacity;
    value_type conductivity;
    value_type density;
    value_type viscosity;
    value_type d_capacity_d_T;
    value_type d_conductivity_d_T;
    value_type d_density_d_T;
    value_type gas_fraction;
    value_type liquid_fraction;
    value_type solid_fraction;
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
      , gas(data.first)
      , liquid(data.second)
      , solid(data.solid)
      , material_type(material_type) // TODO determine material type depending on MaterialData
    {}

    /**
     * Return a reference to the material MaterialData.
     */
    const MaterialData<number> &
    get_data() const
    {
      return data;
    }

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
     * This overload of compute_parameters() can be used only in the context of two-phase
     * simulations (gas-liquid or liquid-solid).
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
     * This overload of compute_parameters() can be used only in the context of three-phase
     * simulations (gas-liquid-solid).
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
     * Get arbitrary parameters, specified by @p flags, for the currently active material_type.
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
              return gas;
            }
          case MaterialTypes::gas_liquid:
            case MaterialTypes::gas_liquid_consistent_with_evaporation: {
              const MaterialParameterValues<value_type> &g = gas;
              const MaterialParameterValues<value_type> &l = liquid;

              const auto &level_set_heaviside = v1;

              if (flags & MaterialUpdateFlags::capacity)
                t.capacity = compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                                          g.capacity,
                                                                          l.capacity);
              if (flags & MaterialUpdateFlags::conductivity)
                t.conductivity = compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                                              g.conductivity,
                                                                              l.conductivity);
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
              if (flags & MaterialUpdateFlags::viscosity)
                t.viscosity = compute_two_phase_fluid_property<value_type>(level_set_heaviside,
                                                                           g.viscosity,
                                                                           l.viscosity);
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
              const MaterialParameterValues<value_type> &l = liquid;
              const MaterialParameterValues<value_type> &s = solid;

              const auto &temperature = v1;
              const auto  temperature_dependent_solid_fraction =
                compute_temperature_dependent_solid_fraction(temperature);

              if (flags & MaterialUpdateFlags::capacity)
                t.capacity = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction, l.capacity, s.capacity);
              if (flags & MaterialUpdateFlags::conductivity)
                t.conductivity = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction, l.conductivity, s.conductivity);
              if (flags & MaterialUpdateFlags::density)
                t.density = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction, l.density, s.density);
              if (flags & MaterialUpdateFlags::viscosity)
                t.viscosity = compute_solid_liquid_phases_property<value_type>(
                  temperature_dependent_solid_fraction, l.viscosity, s.viscosity);
              if (flags & MaterialUpdateFlags::d_capacity_d_T)
                t.d_capacity_d_T =
                  compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                    temperature_dependent_solid_fraction, l.capacity, s.capacity);
              if (flags & MaterialUpdateFlags::d_conductivity_d_T)
                t.d_conductivity_d_T =
                  compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                    temperature_dependent_solid_fraction, l.conductivity, s.conductivity);
              if (flags & MaterialUpdateFlags::d_density_d_T)
                t.d_density_d_T =
                  compute_temperature_derivative_of_solid_liquid_phases_property<value_type>(
                    temperature_dependent_solid_fraction, l.density, s.density);
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
              const MaterialParameterValues<value_type> &g = gas;
              const MaterialParameterValues<value_type> &l = liquid;
              const MaterialParameterValues<value_type> &s = solid;

              const auto &level_set_heaviside = v1;
              const auto &temperature         = v2;
              const auto  temperature_dependent_solid_fraction =
                compute_temperature_dependent_solid_fraction(temperature);

              if (flags & MaterialUpdateFlags::capacity)
                t.capacity = compute_solid_liquid_gas_phases_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  g.capacity,
                  l.capacity,
                  s.capacity);
              if (flags & MaterialUpdateFlags::conductivity)
                t.conductivity = compute_solid_liquid_gas_phases_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  g.conductivity,
                  l.conductivity,
                  s.conductivity);
              if (flags & MaterialUpdateFlags::density)
                {
                  if (material_type == MaterialTypes::gas_liquid_solid_consistent_with_evaporation)
                    t.density = compute_solid_liquid_gas_phases_density_consistent_with_evaporation<
                      value_type>(level_set_heaviside,
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
              if (flags & MaterialUpdateFlags::viscosity)
                t.viscosity = compute_solid_liquid_gas_phases_property<value_type>(
                  level_set_heaviside,
                  temperature_dependent_solid_fraction,
                  g.viscosity,
                  l.viscosity,
                  s.viscosity);
              if (flags & MaterialUpdateFlags::d_capacity_d_T)
                t.d_capacity_d_T =
                  compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                    level_set_heaviside,
                    temperature_dependent_solid_fraction,
                    l.capacity,
                    s.capacity);
              if (flags & MaterialUpdateFlags::d_conductivity_d_T)
                t.d_conductivity_d_T =
                  compute_temperature_derivative_of_solid_liquid_gas_property<value_type>(
                    level_set_heaviside,
                    temperature_dependent_solid_fraction,
                    l.conductivity,
                    s.conductivity);
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
     * Determine a material parameter for two phase flow. If level_set_heaviside = 0, the
     * parameter results in the @p gas_value and if @p ls_heaviside_val = 1 it results
     * in the @p liquid_solid_value. Across the interface, the parameter jumps if
     * TwoPhaseFluidPropertiesTransitionType is "sharp". Otherwise the parameter x is
     * distributed according to the level set function
     *
     * x = (1-ls) * x_g + ls * x_l
     *
     * with the heaviside representation of the level set function ls (level_set_heaviside),
     * the value of the gaseous phase x_g (@p gas_value) and the value of the liquid (and
     * solid) phase x_l (@p liquid_solid_value).
     *
     * @note In case of a material containing solid/liquid/gas phases, the value of
     *       @p liquid_solid_value represents the liquid/solid phase's (level set = 1)
     *       value, as determined by compute_solid_liquid_phases_property().
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
     * level_set_heaviside = 0 this function returns the @p gas_density and if
     * @p ls_heaviside_val = 1 it returns the @p liquid_solid_density. Across the interface, the density
     * is distributed following a reciprocal distribution function
     *
     *  1     ls      (1-ls)
     * --- =  ---- + --------
     *  x     x_l      x_g
     *
     * with the heaviside representation of the level set function ls (@p level_set_heaviside),
     * the density of the gaseous phase x_g (@p gas_density) and the density of the liquid (and
     * solid) phase x_l (@p liquid_solid_density).
     *
     * @note In case of a material containing solid/liquid/gas phases, the value of
     *       @p liquid_solid_value represents the liquid/solid phase's (level set = 1)
     *       value, as determined by compute_solid_liquid_phases_property().
     */
    template <typename value_type>
    inline value_type
    compute_two_phase_fluid_density_consistent_with_evaporation(
      const value_type &level_set_heaviside,
      const value_type &gas_density,
      const value_type &liquid_solid_density) const
    {
      // clang-format off
      return 1.
             / // -------------------------------------------------------------------------------------
             (level_set_heaviside / liquid_solid_density + (1. - level_set_heaviside) / gas_density);
      // clang-format on
    }

    /**
     * Determine a material parameter for the solid/liquid phases. In the mushy zone (where
     * the solid_fraction is between 0 and 1) the material parameter will be interpolated
     * by a smooth cubic spline function, see UtilityFunctions::interpolate_cubic(),
     *
     * x = x_l + (x_s - x_l) * (-2*sf^3 + 3*sf^2)
     *
     * with the solid_fraction sf (@p temperature_dependent_solid_fraction), the value of
     * the solid phase x_s (@p solid_value) and the value of the liquid phase x_l (@p liquid_value).
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
     * with the solid_fraction sf (@p temperature_dependent_solid_fraction), the value of the
     * solid phase x_s (@p solid_value), the value of the liquid phase x_l (@p liquid_value),
     * the liquidus temperature T_liq and the solidus temperature T_sol.
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
     * Determine the parameter of a material containing solid/liquid/gas phases.
     *
     *                         /                           \
     * x = (1-ls) * x_g + ls * | (1 - sf) * x_l + sf * x_s |
     *                         \                           /
     *
     *                        ^-----------------------------^
     *                                      (1)
     *     ^------------------------------------------------^
     *                        (2)
     *
     * with the heaviside representation of the level set function ls (@p level_set_heaviside) and
     * the solid fraction sf (@p temperature_dependent_solid_fraction).
     * (1) is calculated using compute_solid_liquid_phases_property() and is plugged into
     * compute_two_phase_fluid_property() (2) reulting in the parameter x.
     *
     * @p gas_value The parameter's value for the gaseous phase
     * @p liquid_value The parameter's value for the liquid phase
     * @p solid_value The parameter's value for the solid phase
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
     * Determine the density x of a material containing solid/liquid/gas phases consistently with
     * mass flux due to evaporation. Across the interface, the density is distributed following a
     * reciprocal distribution function
     *
     *  1                 ls                   (1-ls)
     * --- =  ----------------------------- + --------
     *  x      ___________________________      x_g
     *        |                           |
     *    (1) | (sf * x_s + (1-sf) * x_l) |
     *        |___________________________|
     *
     *
     *       ^-----------------------------------------^
     *                           (2)
     *
     * with the heaviside representation of the level set function ls (@p level_set_heaviside) and
     * the solid fraction sf (@p temperature_dependent_solid_fraction).
     * (1) is calculated using compute_solid_liquid_phases_property() and is plugged into
     * compute_two_phase_fluid_density_consistent_with_evaporation() (2) resulting in the parameter
     * x.
     *
     * @p gas_value The parameter's value for the gaseous phase
     * @p liquid_value The parameter's value for the liquid phase
     * @p solid_value The parameter's value for the solid phase
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
     * Determine the derivative of a material parameter with respect to the temperature for
     * a material containing solid/liquid/gaseous phases.
     * This function returns the temperature derivatives of the values determined by
     * compute_solid_liquid_gas_phases_property(). Across the interface, the derivative
     * reduces to zero if TwoPhaseFluidPropertiesTransitionType is sharp. Otherwise the
     * derivative d_x/d_T is computed by
     *
     *  dx          dx_ls
     * ---- = ls * -------
     *  dT           dT
     *
     * with the heaviside representation of the level set function ls (level_set_heaviside),
     * the derivative of the liquid(and solid) phase x_ls, which is determined by
     * compute_temperature_derivative_of_solid_liquid_phases_property().
     *
     * @note The gaseous phases is identified based on the level set value only. Thus, its
     *       temperature derivative is zero.
     *
     * @p liquid_value The parameter's value for the liquid phase
     * @p solid_value The parameter's value for the solid phase
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
     * Determine the derivative of the density with respect to the temperature for
     * a material containing solid/liquid/gaseous phases consistent with the
     * mass flux due to evaporation. This function returns the temperature derivatives
     * of the values determined by
     * compute_solid_liquid_gas_phases_density_consistent_with_evaporation().
     *
     * The derivative dx_d_T of the solid/liquid/gaseous phases material is computed by
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
     * @p gas_value The parameter's value for the gaseous phase
     * @p liquid_value The parameter's value for the liquid phase
     * @p solid_value The parameter's value for the solid phase
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
     * Compute the solid fraction depending on the SolidLiquidPropertiesTransitionType.
     *
     * SolidLiquidPropertiesTransitionType = mushy_zone
     *   If the temperature is greater than the liquidus temperature, then the solid fraction is
     *   zero. If the temperature is less than the solidus temperature, then the solid fraction is
     *   one. In between there is a linear interpolation.
     *
     * SolidLiquidPropertiesTransitionType = sharp
     *   The solid fraction is zero above the melting point and one below the melting point.
     *
     * @note The computation of the solid fraction within this function is performed solely
     *       based on the temperature. In the case of a solid/liquid/gaseous material,
     *       thus the gaseous phase is ignored.
     */
    inline number
    compute_temperature_dependent_solid_fraction(const number temperature) const
    {
      if (data.solidification_type == SolidLiquidPropertiesTransitionType::mushy_zone)
        return UtilityFunctions::limit_to_bounds(
          (data.liquidus_temperature - temperature) * data.inv_mushy_interval, 0.0, 1.0);
      else if (data.solidification_type == SolidLiquidPropertiesTransitionType::sharp)
        return temperature < data.melting_point ? 1.0 : 0.0;
      Assert(false, ExcNotImplemented());
      return 0.0;
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
      Assert(false, ExcNotImplemented());
      return VectorizedArray<number>(0.0);
    }

    struct MaterialParameterValuesContainer
    {
      template <typename material_phase_data_struct>
      MaterialParameterValuesContainer(const material_phase_data_struct data)
        : scalar_parameters(data)
        , vectorized_parameters(data)
      {}

      const MaterialParameterValues<number>                          scalar_parameters;
      const MaterialParameterValues<dealii::VectorizedArray<number>> vectorized_parameters;

      constexpr operator const MaterialParameterValues<number> &() const
      {
        return scalar_parameters;
      }

      constexpr operator const MaterialParameterValues<VectorizedArray<number>> &() const
      {
        return vectorized_parameters;
      }
    };

    const MaterialData<number> &data;

    MaterialParameterValuesContainer gas;
    MaterialParameterValuesContainer liquid;
    MaterialParameterValuesContainer solid;

    const MaterialTypes material_type;
  };
} // namespace MeltPoolDG
