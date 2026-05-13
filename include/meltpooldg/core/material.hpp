#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/heat/apparent_capacity.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

#include <vector>

namespace MeltPoolDG
{
  template <typename value_type, typename number>
  struct MaterialParameterValues
  {
    MaterialParameterValues(const number thermal_conductivity_in                = 0.0,
                            const number specific_heat_capacity_in              = 0.0,
                            const number density_in                             = 0.0,
                            const number dynamic_viscosity_in                   = 0.0,
                            const number volume_specific_heat_capacity_in       = 0.0,
                            const number d_thermal_conductivity_d_T_in          = 0.0,
                            const number d_specific_heat_capacity_d_T_in        = 0.0,
                            const number d_density_d_T_in                       = 0.0,
                            const number d_volume_specific_heat_capacity_d_T_in = 0.0,
                            const number gas_fraction_in                        = 0.0,
                            const number liquid_fraction_in                     = 0.0,
                            const number solid_fraction_in                      = 0.0);

    template <typename material_phase_data_struct>
    MaterialParameterValues(const material_phase_data_struct &data);

    value_type thermal_conductivity;
    value_type specific_heat_capacity;
    value_type density;
    value_type dynamic_viscosity;
    value_type volume_specific_heat_capacity;
    value_type d_thermal_conductivity_d_T;
    value_type d_specific_heat_capacity_d_T;
    value_type d_density_d_T;
    value_type d_volume_specific_heat_capacity_d_T;
    value_type gas_fraction;
    value_type liquid_fraction;
    value_type solid_fraction;
  };

  enum class MaterialTypes
  {
    gas,
    liquid,
    gas_liquid,
    gas_liquid_consistent_with_evaporation,
    liquid_solid,
    gas_liquid_solid,
    gas_liquid_solid_consistent_with_evaporation
  };

  MaterialTypes
  determine_material_type(const bool do_two_phase,
                          const bool do_solidification,
                          const bool do_evaporation);

  namespace MaterialUpdateFlags
  {
    enum MaterialUpdateFlags
    {
      none                                = 0,
      thermal_conductivity                = 1 << 0,
      specific_heat_capacity              = 1 << 1,
      density                             = 1 << 2,
      dynamic_viscosity                   = 1 << 3,
      volume_specific_heat_capacity       = 1 << 4,
      d_thermal_conductivity_d_T          = 1 << 5,
      d_specific_heat_capacity_d_T        = 1 << 6,
      d_density_d_T                       = 1 << 7,
      d_volume_specific_heat_capacity_d_T = 1 << 8,
      phase_fractions                     = 1 << 9
    };

    inline MaterialUpdateFlags
    operator|(const MaterialUpdateFlags f1, const MaterialUpdateFlags f2);

    inline MaterialUpdateFlags &
    operator|=(MaterialUpdateFlags &f1, const MaterialUpdateFlags f2);


    inline MaterialUpdateFlags
    operator&(const MaterialUpdateFlags f1, const MaterialUpdateFlags f2);

    inline MaterialUpdateFlags &
    operator&=(MaterialUpdateFlags &f1, const MaterialUpdateFlags f2);

  } // namespace MaterialUpdateFlags

  template <typename number>
  class Material
  {
  public:
    enum FieldType
    {
      none,
      temperature,
      level_set
    };

    Material(const MaterialData<number> &material_data, const MaterialTypes material_type);

    /**
     * Return a reference to the material MaterialData.
     *
     * @note This function does not compute multiphase material parameters. Only use this
     * function if you need access to the raw MaterialData struct.
     */
    const MaterialData<number> &
    get_data() const;

    /**
     * This overload of compute_parameters() can be used only in the context of single phase
     * simulations.
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type, number>
    compute_parameters(const MaterialUpdateFlags::MaterialUpdateFlags &flags) const;

    /**
     * This overload of compute_parameters() can be used only in the context of two-phase
     * simulations (gas-liquid or liquid-solid).
     *
     * If the material_type is MaterialTypes::gas_liquid, @p v is the heaviside representation of the
     * level set, if the material_type is MaterialTypes::liquid_solid, @p v is the temperature.
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type, number>
    compute_parameters(const value_type                               &v,
                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const;

    /**
     * This overload of compute_parameters() can be used only in the context of three-phase
     * simulations (gas-liquid-solid).
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type, number>
    compute_parameters(const value_type                               &level_set_heaviside,
                       const value_type                               &temperature,
                       const MaterialUpdateFlags::MaterialUpdateFlags &flags) const;

    /**
     * This overload of compute_parameters() can be used for any material_type.
     *
     * The @p level_set_heaviside_val and @p temperature_val FECellIntegrators are evaluated only if
     * the respective values are required for the current material_type. I.e. for
     *
     * - MaterialTypes::single_phase,
     *     neither are evaluated.
     * - MaterialTypes::gas_liquid and
     *   MaterialTypes::gas_liquid_consistent_with_evaporation
     *     only @p level_set_heaviside_val is evaluated.
     * - MaterialTypes::liquid_solid
     *     only @p temperature_val
     * - MaterialTypes::gas_liquid_solid and
     *   MaterialTypes::gas_liquid_solid_consistent_with_evaporation
     *     both are evaluated.
     *
     * Before the evaluation of the respective FECellIntegrator's, the values must be set via
     * FEEvaluation::evaluate() with EvaluationFlags::values, or via
     * FEEvaluationBase::submit_value(). The values are evaluated at the quadrature point number @p q_index.
     */
    template <typename value_type, int dim>
    inline MaterialParameterValues<value_type, number>
    compute_parameters(const FECellIntegrator<dim, 1, number>         &level_set_heaviside_val,
                       const FECellIntegrator<dim, 1, number>         &temperature_val,
                       const MaterialUpdateFlags::MaterialUpdateFlags &flags,
                       const unsigned int                              q_index) const;

    /**
     * Same as above, but with the ability to handle a cut temperature.
     *
     * The FECellIntegrator for the temperature is wrapped in a std::vector. If that vector has two
     * entries, the cell is assumed to be an intersected cell and the first entry is the liquid side
     * and the second entry is the gas side. Then we use the heaviside level set as an indicator to
     * select the correct temperature value for each quadrature point.
     */
    template <typename value_type, int dim>
    inline MaterialParameterValues<value_type, number>
    compute_parameters(const FECellIntegrator<dim, 1, number>              &level_set_heaviside_val,
                       const std::vector<FECellIntegrator<dim, 1, number>> &temperature_val,
                       const MaterialUpdateFlags::MaterialUpdateFlags      &flags,
                       const unsigned int                                   q_index) const;

    /**
     * Check whether the material type depends on a certain field variable.
     *
     * @param field_type Potentially dependent field variable.
     */
    bool
    has_dependency(const FieldType &field_type) const;

  private:
    /**
     * Get arbitrary parameters, specified by @p flags, for the currently active material_type.
     */
    template <typename value_type>
    inline MaterialParameterValues<value_type, number>
    compute_parameters_internal(const value_type                               &v1,
                                const value_type                               &v2,
                                const MaterialUpdateFlags::MaterialUpdateFlags &flags) const;

    /**
     * Determine a material parameter for two phase flow. If level_set_heaviside = 0, the
     * parameter results in the @p gas_value and if @p ls_heaviside_val = 1 it results
     * in the @p liquid_solid_value. Across the interface, the parameter jumps if
     * TwoPhaseFluidPropertiesTransitionType is "sharp". Otherwise, the parameter x is
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
                                     const value_type &liquid_solid_value) const;

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
      const value_type &liquid_solid_density) const;

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
                                         const value_type &solid_value) const;

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
      const value_type &solid_value) const;

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
                                             const value_type &solid_value) const;

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
      const value_type &solid_value) const;

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
      const value_type &solid_value) const;

    /**
     * Compute the effective specific heat capacity for a two-phase
     * solid-liquid material.
     *
     * The solid and liquid phase values are interpolated using the
     * temperature-dependent solid fraction. If an apparent heat capacity model
     * is enabled, its latent-heat contribution is added to the interpolated
     * phase value.
     *
     * @tparam value_type Scalar or vectorized value type.
     *
     * @param temperature_dependent_solid_fraction Temperature-dependent solid
     *        fraction used for the solid-liquid interpolation.
     * @param liquid_value Specific heat capacity of the liquid phase.
     * @param solid_value Specific heat capacity of the solid phase.
     *
     * @return Effective solid-liquid specific heat capacity, including the
     *         apparent heat capacity contribution if enabled.
     */
    template <typename value_type>
    inline value_type
    compute_solid_liquid_phases_specific_heat_capacity(
      const value_type &temperature_dependent_solid_fraction,
      const value_type &liquid_value,
      const value_type &solid_value) const;

    /**
     * Compute the effective specific heat capacity for a three-phase
     * solid-liquid-gas material.
     *
     * The solid and liquid phase values are interpolated using the
     * temperature-dependent solid fraction. If an apparent heat capacity model
     * is enabled, its latent-heat contribution is added to the solid-liquid
     * contribution before blending with the gas phase through the level-set
     * Heaviside value.
     *
     * @tparam value_type Scalar or vectorized value type.
     *
     * @param level_set_heaviside Heaviside value used to interpolate between
     *        the gas phase and the condensed solid-liquid phases.
     * @param temperature_dependent_solid_fraction Temperature-dependent solid
     *        fraction used for the solid-liquid interpolation.
     * @param gas_value Specific heat capacity of the gas phase.
     * @param liquid_value Specific heat capacity of the liquid phase.
     * @param solid_value Specific heat capacity of the solid phase.
     *
     * @return Effective solid-liquid-gas specific heat capacity, including the
     *         apparent heat capacity contribution if enabled.
     */
    template <typename value_type>
    inline value_type
    compute_solid_liquid_gas_phases_specific_heat_capacity(
      const value_type &level_set_heaviside,
      const value_type &temperature_dependent_solid_fraction,
      const value_type &gas_value,
      const value_type &liquid_value,
      const value_type &solid_value) const;

    /**
     * Compute the temperature derivative of the effective specific heat
     * capacity for a two-phase solid-liquid material.
     *
     * The derivative contains the contribution from the temperature-dependent
     * solid-liquid interpolation. If an apparent heat capacity model is
     * enabled, the temperature derivative of the apparent heat capacity is
     * added.
     *
     * @tparam value_type Scalar or vectorized value type.
     *
     * @param temperature_dependent_solid_fraction Temperature-dependent solid
     *        fraction used for the solid-liquid interpolation.
     * @param liquid_value Specific heat capacity of the liquid phase.
     * @param solid_value Specific heat capacity of the solid phase.
     *
     * @return Temperature derivative of the effective solid-liquid specific
     *         heat capacity, including the apparent heat capacity derivative if
     *         enabled.
     */
    template <typename value_type>
    inline value_type
    compute_temperature_derivative_of_solid_liquid_specific_heat_capacity(
      const value_type &temperature_dependent_solid_fraction,
      const value_type &liquid_value,
      const value_type &solid_value) const;

    /**
     * Compute the temperature derivative of the effective specific heat
     * capacity for a three-phase solid-liquid-gas material.
     *
     * The derivative contains the contribution from the temperature-dependent
     * solid-liquid interpolation and, if enabled, the temperature derivative of
     * the apparent heat capacity. The gas phase is assumed to be independent of
     * temperature in this derivative contribution, while the condensed phase
     * contribution is blended with the gas phase through the level-set
     * Heaviside value.
     *
     * @tparam value_type Scalar or vectorized value type.
     *
     * @param level_set_heaviside Heaviside value used to interpolate between
     *        the gas phase and the condensed solid-liquid phases.
     * @param temperature_dependent_solid_fraction Temperature-dependent solid
     *        fraction used for the solid-liquid interpolation.
     * @param liquid_value Specific heat capacity of the liquid phase.
     * @param solid_value Specific heat capacity of the solid phase.
     *
     * @return Temperature derivative of the effective solid-liquid-gas specific
     *         heat capacity, including the apparent heat capacity derivative if
     *         enabled.
     */
    template <typename value_type>
    inline value_type
    compute_temperature_derivative_of_solid_liquid_gas_specific_heat_capacity(
      const value_type &level_set_heaviside,
      const value_type &temperature_dependent_solid_fraction,
      const value_type &liquid_value,
      const value_type &solid_value) const;

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
      const value_type &solid_value) const;

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
    compute_temperature_dependent_solid_fraction(const number temperature) const;

    inline dealii::VectorizedArray<number>
    compute_temperature_dependent_solid_fraction(
      const dealii::VectorizedArray<number> &temperature) const;

    struct MaterialParameterValuesContainer
    {
      template <typename material_phase_data_struct>
      MaterialParameterValuesContainer(const material_phase_data_struct data);

      const MaterialParameterValues<number, number>                          scalar_parameters;
      const MaterialParameterValues<dealii::VectorizedArray<number>, number> vectorized_parameters;

      constexpr operator const MaterialParameterValues<number, number> &() const;

      constexpr
      operator const MaterialParameterValues<dealii::VectorizedArray<number>, number> &() const;
    };

    const MaterialData<number> &data;

    MaterialParameterValuesContainer gas;
    MaterialParameterValuesContainer liquid;
    MaterialParameterValuesContainer solid;

    const MaterialTypes material_type;

    const number inv_mushy_interval;

    std::unique_ptr<Heat::ApparentCapacity<number>> apparent_capacity = nullptr;
  };


  /*----------------- Template constructors -----------------*/


  template <typename value_type, typename number>
  MaterialParameterValues<value_type, number>::MaterialParameterValues(
    const number thermal_conductivity_in,
    const number specific_heat_capacity_in,
    const number density_in,
    const number dynamic_viscosity_in,
    const number volume_specific_heat_capacity_in,
    const number d_thermal_conductivity_d_T_in,
    const number d_specific_heat_capacity_d_T_in,
    const number d_density_d_T_in,
    const number d_volume_specific_heat_capacity_d_T_in,
    const number gas_fraction_in,
    const number liquid_fraction_in,
    const number solid_fraction_in)
    : thermal_conductivity(thermal_conductivity_in)
    , specific_heat_capacity(specific_heat_capacity_in)
    , density(density_in)
    , dynamic_viscosity(dynamic_viscosity_in)
    , volume_specific_heat_capacity(volume_specific_heat_capacity_in)
    , d_thermal_conductivity_d_T(d_thermal_conductivity_d_T_in)
    , d_specific_heat_capacity_d_T(d_specific_heat_capacity_d_T_in)
    , d_density_d_T(d_density_d_T_in)
    , d_volume_specific_heat_capacity_d_T(d_volume_specific_heat_capacity_d_T_in)
    , gas_fraction(gas_fraction_in)
    , liquid_fraction(liquid_fraction_in)
    , solid_fraction(solid_fraction_in)
  {}



  template <typename value_type, typename number>
  template <typename material_phase_data_struct>
  MaterialParameterValues<value_type, number>::MaterialParameterValues(
    const material_phase_data_struct &data)
    : thermal_conductivity(data.thermal_conductivity)
    , specific_heat_capacity(data.specific_heat_capacity)
    , density(data.density)
    , dynamic_viscosity(data.dynamic_viscosity)
    , volume_specific_heat_capacity(data.density * data.specific_heat_capacity)
    , d_thermal_conductivity_d_T(0.0)
    , d_specific_heat_capacity_d_T(0.0)
    , d_density_d_T(0.0)
    , d_volume_specific_heat_capacity_d_T(0.0)
    , gas_fraction(1.0)
    , liquid_fraction(0.0)
    , solid_fraction(0.0)
  {}



  template <typename number>
  Material<number>::Material(const MaterialData<number> &material_data,
                             const MaterialTypes         material_type)
    : data(material_data)
    , gas(data.gas)
    , liquid(data.liquid)
    , solid(data.solid)
    , material_type(material_type)
    , inv_mushy_interval(material_type == MaterialTypes::liquid_solid ||
                             material_type == MaterialTypes::gas_liquid_solid ||
                             material_type ==
                               MaterialTypes::gas_liquid_solid_consistent_with_evaporation ?
                           1.0 / (data.liquidus_temperature - data.solidus_temperature) :
                           0.0)
  {
    if (material_data.latent_heat_of_fusion > 0)
      apparent_capacity = std::make_unique<Heat::ApparentCapacity<number>>(material_data);
  }



  template <typename number>
  template <typename material_phase_data_struct>
  Material<number>::MaterialParameterValuesContainer::MaterialParameterValuesContainer(
    const material_phase_data_struct data)
    : scalar_parameters(data)
    , vectorized_parameters(data)
  {}
} // namespace MeltPoolDG
