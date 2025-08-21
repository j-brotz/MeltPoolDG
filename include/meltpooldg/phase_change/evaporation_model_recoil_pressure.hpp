#pragma once

#include <deal.II/base/vectorization.h>

#include <meltpooldg/phase_change/evaporation_model_base.hpp>
#include <meltpooldg/phase_change/recoil_pressure_data.hpp>
#include <meltpooldg/phase_change/recoil_pressure_operation.hpp>

namespace MeltPoolDG::Evaporation
{
  /**
   * This class implements the evaporative mass flux computed from the
   * recoil pressure as presented in, e.g.,
   *
   * S.A. Khairallah, A.T. Anderson, A. Rubenchik, W.E. King,
   * Laser powder-bed fusion additive manufacturing: Physics of complex
   * melt flow and formation mechanisms of pores, spatter, and denudation
   * zones, Acta Mater. 108 (2016) 36–45.
   * DOI: 10.1016/j.actamat.2016.02.014.
   *
   */
  template <typename number>
  class EvaporationModelRecoilPressure : public EvaporationModelBase<number>
  {
  private:
    const RecoilPressurePhenomenologicalModel<number> recoil_model;

    // according to Meier 2020
    const number sticking_constant;
    const number Cm;                   // molar_mass/(2*pi*molar_gas_constant)
    const number temperature_constant; // molar_latent_heat_of_evaporation / molar_gas_constant

  public:
    EvaporationModelRecoilPressure(const RecoilPressureData<number> &recoil_data,
                                   const number                      boiling_temperature,
                                   const number                      molar_mass,
                                   const number                      latent_heat_evaporation);

    /**
     * The evaporative mass flux is computed as
     *
     *                                  ______
     *                                 /  M
     * .                              /-------
     * m(T) = 0.82 · c_s  · p_v(T) · √  2πR T
     *
     * where c_s is the sticking coefficient, p_v(T) the recoil pressure,
     * M the molar mass, R the molar gas constant and T the temperature.
     */
    number
    local_compute_evaporative_mass_flux(const number T) const final;

    dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec(const dealii::VectorizedArray<number> &T) const final;

    /**
     * Compute the derivative of the evaporative mass flux as
     *
     *    .
     *  d m                 .    /  c_T       1  \
     * ----- = 0.82 · c_s · m(T) | ----- - ----- |
     *  d T                      \   T²     2 T  /
     *
     * where it is assumed that the recoil pressure p_v(T) is computed as
     *
     *                        /       /  1       1  \\
     *  p_v(T) = s * c_p * exp|-c_T * | ---  -  --- ||
     *                        \       \  T      T_v //
     *
     * according to the phenomenological model by Anisimov and Khokhlov as implemented in
     * RecoilPressurePhenomenologicalModel.
     *
     * S. Anisimov and V. Khokhlov. Instabilities in Laser-Matter Interaction. CRC Press, Boca
     * Raton, FL, 1995.
     *
     */
    number
    local_compute_evaporative_mass_flux_derivative(const number T) const final;

    dealii::VectorizedArray<number>
    local_compute_evaporative_mass_flux_vec_derivative(
      const dealii::VectorizedArray<number> &T) const final;
  };
} // namespace MeltPoolDG::Evaporation
