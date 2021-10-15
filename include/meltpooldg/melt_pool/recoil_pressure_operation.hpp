/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, February 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
// MeltPoolDG
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::MeltPool
{
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

    const RecoilPressureData<double> &recoil_pressure_data;

    const double boiling_temperature;
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

  public:
    RecoilPressureOperation(const ScratchData<dim> &  scratch_data_in,
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
    compute_recoil_pressure_force(VectorType &      force_rhs,
                                  const VectorType &level_set_as_heaviside,
                                  const VectorType &temperature,
                                  bool              zero_out = true) const;

    /**
     * compute the recoil pressure coefficient depending on the temperature
     */
    static double
    compute_recoil_pressure_coefficient(const double  T,
                                        const double &pressure_constant,
                                        const double &temperature_constant,
                                        const double &boiling_temperature);

  private:
    /**
     * compute the recoil pressure coefficient depending on the temperature
     */
    inline double
    compute_recoil_pressure_coefficient(const double T) const;
  };
} // namespace MeltPoolDG::MeltPool
