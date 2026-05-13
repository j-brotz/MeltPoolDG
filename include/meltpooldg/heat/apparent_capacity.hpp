
#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/core/material_data.hpp>

namespace MeltPoolDG::Heat
{
  template <typename number>
  struct QLQParameters
  {
    QLQParameters(const MaterialData<number> &material)
      : dT(material.liquidus_temperature - material.solidus_temperature)
      , T_peak(0.5 * (material.solidus_temperature + material.liquidus_temperature))
      , C_peak(2.0 * material.latent_heat_of_fusion / dT)
      , delta(dT / 6.0)
    {}
    /// Width of the mushy zone.
    ///
    /// Defined as the temperature interval between liquidus and solidus
    /// temperature.
    const number dT;

    /// Temperature at the maximum of the apparent heat capacity contribution.
    ///
    /// For the QLQ apparent capacity model, this is chosen as the midpoint
    /// between the solidus and liquidus temperatures.
    const number T_peak;

    /// Peak value of the QLQ apparent heat capacity contribution.
    ///
    /// The factor of two ensures that the integrated apparent capacity over the
    /// mushy zone corresponds to the latent heat of fusion.
    const number C_peak;

    /// Width of the quadratic transition region used by the QLQ interpolation.
    ///
    /// This controls the size of the zero-slope smoothing regions close to the
    /// solidus, liquidus, and peak temperatures.
    const number delta;
  };

  /**
   * @brief Apparent capacity model for latent heat release/absorption during
   * phase change.
   *
   * This class evaluates the additional heat capacity contribution used in the
   * apparent capacity method. The latent heat is distributed over the mushy zone
   * according to the selected apparent-capacity profile.
   *
   * The input argument is the solid fraction @f$f_s@f$, where
   * @f$f_s = 1@f$ denotes fully solid material and @f$f_s = 0@f$ denotes fully
   * liquid material.
   *
   * @tparam number Scalar number type.
   */
  template <typename number>
  class ApparentCapacity
  {
  public:
    /**
     * @brief Constructor.
     *
     * Initializes the apparent capacity model from the material data.
     *
     * @param material_data_in Material data containing latent heat, solidus and
     * liquidus temperatures, and the selected apparent-capacity type.
     */
    ApparentCapacity(const MaterialData<number> &material_data_in);

    /**
     * @brief Evaluate the apparent capacity.
     *
     * @param solid_fraction Solid fraction @f$f_s@f$.
     * @return Apparent capacity contribution.
     */
    dealii::VectorizedArray<number>
    evaluate(const dealii::VectorizedArray<number> &solid_fraction) const;

    /**
     * @brief Evaluate the apparent capacity.
     *
     * @param solid_fraction Solid fraction @f$f_s@f$.
     * @return Apparent capacity contribution.
     */
    number
    evaluate(const number &solid_fraction) const;

    /**
     * @brief Compute the temperature derivative of the apparent capacity.
     *
     * @param solid_fraction Solid fraction @f$f_s@f$.
     * @return Temperature derivative of the apparent capacity contribution.
     */
    dealii::VectorizedArray<number>
    compute_temperature_derivative(const dealii::VectorizedArray<number> &solid_fraction) const;

    /**
     * @brief Compute the temperature derivative of the apparent capacity.
     *
     * @param solid_fraction Solid fraction @f$f_s@f$.
     * @return Temperature derivative of the apparent capacity contribution.
     */
    number
    compute_temperature_derivative(const number &solid_fraction) const;

  private:
    /// Material data used by the apparent capacity model.
    const MaterialData<number> &material;

    /// Inverse radius of the mushy zone in temperature space.
    const number inv_mushy_zone_radius;

    /// Peak value of the apparent-capacity distribution.
    const number peak;

    /// Selected apparent-capacity profile.
    const ApparentCapacityType apparent_capacity_type;

    /// Parameters used by the QLQ apparent capacity model.
    const QLQParameters<number> qlq;

    /**
     * @brief Compute the temperature derivative of the QLQ apparent-capacity
     * profile.
     *
     * @param solid_fraction Solid fraction @f$f_s@f$.
     * @return Temperature derivative of the QLQ apparent-capacity contribution.
     */
    number
    compute_temperature_derivative_qlq_from_solid_fraction(const number &solid_fraction) const;

    /**
     * @brief Evaluate the QLQ apparent-capacity profile.
     *
     * @param solid_fraction Solid fraction @f$f_s@f$.
     * @return QLQ apparent-capacity contribution.
     */
    number
    evaluate_qlq_from_solid_fraction(const number &solid_fraction) const;
  };

} // namespace MeltPoolDG::Heat
