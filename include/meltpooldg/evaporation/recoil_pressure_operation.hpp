/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/evaporation/recoil_pressure_data.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/level_set/delta_approximation_phase_weighted.hpp>
#include <meltpooldg/material/material_data.hpp>

namespace MeltPoolDG::Evaporation
{



  /**
   * Base class for the recoil pressure model
   */
  template <typename number>
  class RecoilPressureModelBase
  {
  public:
    virtual number
    compute_recoil_pressure_coefficient(const number /*T*/) const
    {
      AssertThrow(false, ExcNotImplemented());
    };

    virtual VectorizedArray<number>
    compute_recoil_pressure_coefficient(const VectorizedArray<number> & /*T*/) const
    {
      AssertThrow(false, ExcNotImplemented());
    };

    virtual number
    compute_recoil_pressure_coefficient(const number /*T*/,
                                        const number /*m_dot*/,
                                        const number /*delta_coefficient*/) const
    {
      AssertThrow(false, ExcNotImplemented());
    };

    virtual VectorizedArray<number>
    compute_recoil_pressure_coefficient(const VectorizedArray<number> & /*T*/,
                                        const VectorizedArray<number> & /*m_dot*/,
                                        const VectorizedArray<number> & /*delta_coefficient*/) const
    {
      AssertThrow(false, ExcNotImplemented());
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

    VectorizedArray<number>
    compute_recoil_pressure_coefficient(const VectorizedArray<number> &T) const final;
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

    VectorizedArray<number>
    compute_recoil_pressure_coefficient(
      const VectorizedArray<number> &T,
      const VectorizedArray<number> &m_dot,
      const VectorizedArray<number> &delta_coefficient) const final;
  };

  /**
   * The force contribution of the recoil pressure due to evaporation is computed. The model of
   * S.I. Anisimov and V.A. Khokhlov (1995) is considered. The consideration of any other model
   * is however possible. First, the temperature is updated and second, the recoil pressure is
   * computed.
   */
  template <int dim>
  class RecoilPressureOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    const ScratchData<dim> &scratch_data;
    /*
     * indices for getting DoFHandler<dim> and Quadrature<dim> of relevant subproblems
     */
    const unsigned int flow_vel_dof_idx;
    const unsigned int flow_vel_quad_idx;
    const unsigned int flow_pressure_dof_idx;
    const unsigned int ls_dof_idx;
    const unsigned int temp_dof_idx;

    const bool         do_level_set_pressure_gradient_interpolation;
    FullMatrix<double> ls_to_pressure_grad_interpolation_matrix;

    const RecoilPressureModelType model_type;

    std::unique_ptr<const RecoilPressureModelBase<double>> recoil_pressure_model;

    std::unique_ptr<const LevelSet::DeltaApproximationBase<double>> delta_phase_weighted;

  public:
    RecoilPressureOperation(const ScratchData<dim>   &scratch_data_in,
                            const Parameters<double> &data_in,
                            const unsigned int        flow_vel_dof_idx_in,
                            const unsigned int        flow_vel_quad_idx_in,
                            const unsigned int        flow_pressure_dof_idx_in,
                            const unsigned int        ls_dof_idx_in,
                            const unsigned int        temp_dof_idx_in);

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
