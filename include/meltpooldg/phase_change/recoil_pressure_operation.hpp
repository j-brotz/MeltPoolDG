#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/material_data.hpp>
#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
#include <meltpooldg/phase_change/recoil_pressure_data.hpp>

#include <memory>


namespace MeltPoolDG::Evaporation
{
  /**
   * Base class for the recoil pressure model
   */
  template <typename number>
  class RecoilPressureModelBase
  {
  public:
    virtual ~RecoilPressureModelBase() = default;

    virtual number
    compute_recoil_pressure_coefficient(const number /*T*/) const
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    };

    virtual dealii::VectorizedArray<number>
    compute_recoil_pressure_coefficient(const dealii::VectorizedArray<number> & /*T*/) const
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    };

    virtual number
    compute_recoil_pressure_coefficient(const number /*T*/,
                                        const number /*m_dot*/,
                                        const number /*delta_coefficient*/) const
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    };

    virtual dealii::VectorizedArray<number>
    compute_recoil_pressure_coefficient(
      const dealii::VectorizedArray<number> & /*T*/,
      const dealii::VectorizedArray<number> & /*m_dot*/,
      const dealii::VectorizedArray<number> & /*delta_coefficient*/) const
    {
      AssertThrow(false, dealii::ExcNotImplemented());
    };
  };

  /**
   * This class implements the phenomenological recoil pressure model according to
   *
   * Anisimov, S. I., & Khokhlov, V. A. (1995). Instabilities in laser-matter interaction. CRC
   * press.
   */
  template <typename number>
  class RecoilPressurePhenomenologicalModel : public RecoilPressureModelBase<number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using RecoilPressureModelBase<number>::compute_recoil_pressure_coefficient;

  private:
    const RecoilPressureData<number> &recoil_data;
    const number                      boiling_temperature;
    const number                      molar_mass;
    const number                      latent_heat_evaporation;


  public:
    RecoilPressurePhenomenologicalModel(const RecoilPressureData<number> &recoil_data,
                                        const number                      boiling_temperature,
                                        const number                      molar_mass,
                                        const number                      latent_heat_evaporation);

    /**
     * Compute the recoil pressure p_v(T) in terms of the temperature T
     *
     *                          /       /  1       1  \\
     *    p_v(T) = s * c_p * exp|-c_T * | ---  -  --- ||
     *                          \       \  T      T_v //
     *
     * with a scaling factor s (0<=s<=1), the recoil pressure constant c_p, the temperature
     * constant c_T and the boiling temperature T_v.
     *
     * The scaling factor is introduced to potentially smoothly activate the recoil pressure
     * and avoid a sharp activation. It is computed depending on the activation temperature
     * T_ac of the recoil pressure
     *
     *        -
     *       |     0         if T <= T_ac
     *       |
     *       |  T - T_ac
     *   s = |  --------     if T_ac < T < T_v
     *       |  T_v - T_ac
     *       |
     *       |     1         if T >= T_v
     *        -
     *
     * whereas the T_ac<=T_v.
     */
    number
    compute_recoil_pressure_coefficient(const number T) const final;

    dealii::VectorizedArray<number>
    compute_recoil_pressure_coefficient(const dealii::VectorizedArray<number> &T) const final;
  };

  template <typename number>
  class RecoilPressureHybridModel : public RecoilPressureModelBase<number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using RecoilPressureModelBase<number>::compute_recoil_pressure_coefficient;

  private:
    const RecoilPressureData<number>                 &recoil_data;
    const number                                      boiling_temperature;
    const number                                      density_coeff;
    const RecoilPressurePhenomenologicalModel<number> recoil_phenomenological;

  public:
    RecoilPressureHybridModel(const RecoilPressureData<number> &recoil_data,
                              const MaterialData<number>       &material_data);

    /**
     * Compute the recoil pressure p_v(T) in terms of the temperature @p T
     *
     *                          .  /   1       1   \
     *    p_v(T) = p_pheno(T) - m² | ----- - ----- |
     *                             \  ρ_g     ρ_l  /
     *
     *
     * with the phenomenological recoil pressure p_pheno(T) (see documentation @class
     * RecoilPressurePhenomenologicalModel)
     *                           .
     * the evaporative mass flux m (@p m_dot), the density of the gas (vapor) phase ρ_g and the
     * density of the liquid phase ρ_l.
     *
     * The @p delta_coefficient takes into account potentially different delta function between the
     * evaporative flux term in the continuity equation and the momentum balance equation.
     */
    number
    compute_recoil_pressure_coefficient(const number T,
                                        const number m_dot,
                                        const number delta_coefficient) const final;

    dealii::VectorizedArray<number>
    compute_recoil_pressure_coefficient(
      const dealii::VectorizedArray<number> &T,
      const dealii::VectorizedArray<number> &m_dot,
      const dealii::VectorizedArray<number> &delta_coefficient) const final;
  };


  /**
   * @brief This class implements the recoil pressure model computed with
   * pressure-aware boundary conditions, as presented according to
   *
   * Refined Formulations of Resolved Vapor Flow and Unresolved Recoil Pressure Models for Rapid
   * Evaporation in Metal Additive Manufacturing under Elevated Pressure.
   */
  template <typename number>
  class RecoilPressureModelPressureAware : public RecoilPressureModelBase<number>
  {
    //@todo: to avoid compiler warnings regarding hidden overriden functions
    using RecoilPressureModelBase<number>::compute_recoil_pressure_coefficient;

  public:
    /**
     * @brief Constructor.
     *
     * @param pressure_aware_data Data structure holding data for the pressure-aware recoil pressure calculations.
     * @param boling_temperature Boiling temperature at given build chamber pressure level.
     * @param molar_mass Molar mass.
     * @param latent_heat_of_evaporation Latent heat of evaporation.
     */
    RecoilPressureModelPressureAware(
      const typename RecoilPressureData<number>::PressureAwareData &pressure_aware_data,
      const number                                                  boiling_temperature,
      const number                                                  molar_mass,
      const number                                                  latent_heat_evaporation);

    /**
     * @brief Compute recoil pressure coefficient as
     *
     *             Np-1
     * p(T) = pᵍ +  ∑  Kₚ,ᵢ · (T - Tᵥ(pᵍ))ⁱ⁺²
     *             i=0
     *
     * @param T Melt surface temperature.
     * @return Recoil pressure coefficient.
     */
    number
    compute_recoil_pressure_coefficient(const number T) const final;

    /**
     * Vectorized version of compute_recoil_pressure_coefficient(). See documentation above.
     * @param T Melt surface temperature.
     * @return Recoil pressure coefficient.
     */
    dealii::VectorizedArray<number>
    compute_recoil_pressure_coefficient(const dealii::VectorizedArray<number> &T) const final;

  private:
    /// Data structure holding fitting parameters (Kp) and ambient gas pressure
    const typename RecoilPressureData<number>::PressureAwareData pressure_aware_data;

    /// Boiling temperature at given build chamber pressure level
    const number boiling_temperature;

    /// Molar mass
    const number molar_mass;

    /// Latent heat of evaporation
    const number latent_heat_evaporation;

    /// Fitting parameters for empirical correlations in the free-surface model
    std::vector<number> Kp;

    /// Ambient gas pressure (build chamber pressure)
    const number ambient_gas_pressure;
  };


  /**
   * The force contribution of the recoil pressure due to evaporation is computed. The model of
   * S.I. Anisimov and V.A. Khokhlov (1995) is considered. The consideration of any other model
   * is however possible. First, the temperature is updated and second, the recoil pressure is
   * computed.
   */
  template <int dim, typename number>
  class RecoilPressureOperation
  {
  private:
    using VectorType      = dealii::LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<number>;

    const ScratchData<dim, dim, number> &scratch_data;
    /*
     * indices for getting DoFHandler<dim> and Quadrature<dim> of relevant subproblems
     */
    const unsigned int flow_vel_dof_idx;
    const unsigned int flow_vel_quad_idx;
    const unsigned int flow_pressure_dof_idx;
    const unsigned int ls_dof_idx;
    const unsigned int heat_dof_idx;

    const bool                 do_level_set_pressure_gradient_interpolation;
    dealii::FullMatrix<number> ls_to_pressure_grad_interpolation_matrix;

    const RecoilPressureModelType model_type;

    std::unique_ptr<const RecoilPressureModelBase<number>> recoil_pressure_model;

    std::unique_ptr<const LevelSet::DeltaApproximationBase<number>> delta_phase_weighted;

    // For the one phase cut temperature, we need a dummy temperature value for the gas domain -
    // which doesn't have any temperature information. We use the value that is just below the
    // recoil pressure activation temperature, so the recoil pressure will reliably be zero in that
    // domain.
    const number dummy_temperature;

  public:
    RecoilPressureOperation(const ScratchData<dim, dim, number> &scratch_data_in,
                            const RecoilPressureData<number>    &recoil,
                            const MaterialData<number>          &material,
                            const unsigned int                   flow_vel_dof_idx_in,
                            const unsigned int                   flow_vel_quad_idx_in,
                            const unsigned int                   flow_pressure_dof_idx_in,
                            const unsigned int                   ls_dof_idx_in,
                            const unsigned int                   heat_dof_idx_in);

    /**
     * Compute the contribution of the recoil pressure to the force vector of the Navier-Stokes
     * equations from the given temperature field as a volume force by considering a smooth delta
     * function computed from the level set.
     */
    void
    compute_recoil_pressure_force(VectorType        &force_rhs,
                                  const VectorType  &level_set_as_heaviside,
                                  const VectorType  &temperature,
                                  const VectorType  &evaporative_mass_flux,
                                  const unsigned int evapor_dof_idx,
                                  bool               zero_out = true) const;
  };

} // namespace MeltPoolDG::Evaporation
