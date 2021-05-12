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
    const unsigned int ls_dof_idx;
    const unsigned int temp_dof_idx;


  public:
    RecoilPressureOperation(const ScratchData<dim> &  scratch_data_in,
                            const Parameters<double> &data_in,
                            const unsigned int        flow_vel_dof_idx_in,
                            const unsigned int        flow_vel_quad_idx_in,
                            const unsigned int        ls_dof_idx_in,
                            const unsigned int        temp_dof_idx_in)
      : scratch_data(scratch_data_in)
      , recoil_pressure_data(data_in.recoil)
      , boiling_temperature(data_in.mp.boiling_temperature)
      , flow_vel_dof_idx(flow_vel_dof_idx_in)
      , flow_vel_quad_idx(flow_vel_quad_idx_in)
      , ls_dof_idx(ls_dof_idx_in)
      , temp_dof_idx(temp_dof_idx_in)
    {}

    /**
     * Compute the contribution of the recoil pressure to the force vector of the Navier-Stokes
     * equations from the given temperature field as a volume force by considering a smooth delta
     * function computed from the level set.
     */
    void
    compute_recoil_pressure_force(VectorType &      force_rhs,
                                  const VectorType &level_set_as_heaviside,
                                  const VectorType &temperature,
                                  bool              zero_out = true)
    {
      temperature.update_ghost_values();
      scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
        [&](const auto &matrix_free,
            auto &      force_rhs,
            const auto &level_set_as_heaviside,
            auto        macro_cells) {
          FECellIntegrator<dim, 1, double> level_set(matrix_free, ls_dof_idx, flow_vel_quad_idx);

          FECellIntegrator<dim, dim, double> recoil_pressure(matrix_free,
                                                             flow_vel_dof_idx,
                                                             flow_vel_quad_idx);

          FECellIntegrator<dim, 1, double> temperature_val(matrix_free,
                                                           temp_dof_idx,
                                                           flow_vel_quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              level_set.reinit(cell);
              level_set.gather_evaluate(level_set_as_heaviside, false, true);
              // level_set.evaluate(false, true);
              // level_set.read_dof_values_plain(level_set_as_heaviside);
              // level_set.evaluate(false, true);

              temperature_val.reinit(cell);
              temperature_val.read_dof_values_plain(temperature);
              temperature_val.evaluate(true, false);

              recoil_pressure.reinit(cell);

              for (unsigned int q_index = 0; q_index < recoil_pressure.n_q_points; ++q_index)
                {
                  const auto &t = temperature_val.get_value(q_index);

                  VectorizedArray<double> recoil_pressure_coefficient = 0;

                  for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell);
                       ++v)
                    recoil_pressure_coefficient[v] = compute_recoil_pressure_coefficient(t[v]);

                  recoil_pressure.submit_value(recoil_pressure_coefficient *
                                                 level_set.get_gradient(q_index),
                                               q_index);
                }
              recoil_pressure.integrate_scatter(true, false, force_rhs);
            }
        },
        force_rhs,
        level_set_as_heaviside,
        zero_out);
      temperature.zero_out_ghosts();
    }

    /**
     * compute the recoil pressure coefficient depending on the temperature
     */
    inline double
    compute_recoil_pressure_coefficient(const double T)
    {
      return compute_recoil_pressure_coefficient(T,
                                                 recoil_pressure_data.pressure_constant,
                                                 recoil_pressure_data.temperature_constant,
                                                 boiling_temperature);
    }
    /**
     * compute the recoil pressure coefficient depending on the temperature
     */
    static double
    compute_recoil_pressure_coefficient(const double  T,
                                        const double &pressure_constant,
                                        const double &temperature_constant,
                                        const double &boiling_temperature)
    {
      return pressure_constant *
             std::exp(-temperature_constant * (1. / T - 1. / boiling_temperature));
    }
  };
} // namespace MeltPoolDG::MeltPool
