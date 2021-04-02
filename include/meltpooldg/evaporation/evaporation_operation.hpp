/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;
  /**
   *     This module computes for a given evaporative mass flux $\f\dot{m}\f$ the corresponding
   * interface velocity according to
   *
   *     \f[ \boldsymbol{n}\cfrac{\dot{m}}{\rho} \f]
   *
   *     with the normal vector \f$\boldsymbol{n}\f$, the evaporative mass flux \f$\dot{m}\f$
   *     and the density \f$\rho\f$ as well as the corresponding term in the mass balance
   *     equation of the incompressible Navier-Stokes formulation
   *
   *     \f[ \dot{m}\,(\frac{1}{\rho_l}-\frac{1}{\rho_g})\,\delta \f]
   *
   *     with the delta-function \f$\delta\f$.
   *
   */
  template <int dim>
  class EvaporationOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    std::shared_ptr<const ScratchData<dim>> scratch_data;
    /**
     *  parameters controlling the evaporation
     */
    EvaporationData<double> evaporation_data;
    /**
     * references to solutions needed for the computation
     */
    const VectorType &     level_set_as_heaviside;
    const BlockVectorType &normal_vector;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int normal_dof_idx;
    const unsigned int evapor_vel_dof_idx;
    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int ls_quad_idx;
    const double       tolerance_normal_vector;
    /**
     * evaporative mass flux
     */
    VectorType evaporative_mass_flux;
    /**
     * evaporation velocity at quadrature points
     */
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaporation_velocities;
    /**
     * evaporation velocity due to evaporation and flow
     */
    VectorType evaporation_velocity;

  public:
    EvaporationOperation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                         const VectorType &                             level_set_as_heaviside_in,
                         const BlockVectorType &                        normal_vector_in,
                         std::shared_ptr<SimulationBase<dim>>           base_in,
                         const unsigned int                             normal_dof_idx_in,
                         const unsigned int                             evapor_vel_dof_idx_in,
                         const unsigned int                             ls_hanging_nodes_dof_idx_in,
                         const unsigned int                             ls_quad_idx_in)
      : scratch_data(scratch_data_in)
      , evaporation_data(base_in->parameters.evapor)
      , level_set_as_heaviside(level_set_as_heaviside_in)
      , normal_vector(normal_vector_in)
      , normal_dof_idx(normal_dof_idx_in)
      , evapor_vel_dof_idx(evapor_vel_dof_idx_in)
      , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx_in)
      , ls_quad_idx(ls_quad_idx_in)
      , tolerance_normal_vector(
          std::min(1e-2,
                   std::max(std::pow(10,
                                     UtilityFunctions::get_exponent_power_ten(std::pow(
                                       GridTools::volume<dim>(scratch_data->get_triangulation(),
                                                              scratch_data->get_mapping()),
                                       1. / dim))) *
                              1e-3,
                            1e-12)))
    {
      reinit();
    }

    void
    reinit()
    {
      scratch_data->initialize_dof_vector(
        evaporative_mass_flux, ls_hanging_nodes_dof_idx); // @todo: evapor_dof_idx/temp_dof_idx
      scratch_data->initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);
    }

    void
    compute_evaporative_mass_flux_from_temperature(const VectorType & temperature,
                                                   const unsigned int temp_dof_idx,
                                                   const double &     boiling_temperature,
                                                   const double &     pressure_constant    = 0.0,
                                                   const double &     temperature_constant = 0.0)
    {
      const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(temp_dof_idx);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : scratch_data->get_dof_handler(temp_dof_idx).active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              evaporative_mass_flux[local_dof_indices[i]] =
                compute_temperature_dependent_mass_flux_rate(temperature[local_dof_indices[i]],
                                                             pressure_constant,
                                                             temperature_constant,
                                                             boiling_temperature);
          }
    }

    void
    compute_evaporation_velocity()
    {
      level_set_as_heaviside.update_ghost_values();
      normal_vector.update_ghost_values();
      evaporative_mass_flux.update_ghost_values();

      FECellIntegrator<dim, 1, double> ls(scratch_data->get_matrix_free(),
                                          ls_hanging_nodes_dof_idx,
                                          ls_quad_idx);

      FECellIntegrator<dim, dim, double> normal_vec(scratch_data->get_matrix_free(),
                                                    normal_dof_idx,
                                                    ls_quad_idx);

      FECellIntegrator<dim, 1, double> evap_flux(
        scratch_data->get_matrix_free(),
        ls_hanging_nodes_dof_idx, // @todo: generalize --> temp_dof_idx
        ls_quad_idx);

      evaporation_velocities.resize(scratch_data->get_matrix_free().n_cell_batches() *
                                    ls.n_q_points);

      for (unsigned int cell = 0; cell < scratch_data->get_matrix_free().n_cell_batches(); ++cell)
        {
          Tensor<1, dim, VectorizedArray<double>> *evapor_vel = begin_evaporation_velocity(cell);

          ls.reinit(cell);
          ls.read_dof_values(level_set_as_heaviside);
          ls.evaluate(true, true);

          normal_vec.reinit(cell);
          normal_vec.read_dof_values(normal_vector);
          normal_vec.evaluate(true, false);

          evap_flux.reinit(cell);
          evap_flux.read_dof_values(evaporative_mass_flux);
          evap_flux.evaluate(true, false);

          for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
            {
              const auto n_phi =
                MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index),
                                                        tolerance_normal_vector);

              evapor_vel[q_index] = n_phi * evap_flux.get_value(q_index) *
                                    (ls.get_value(q_index) / evaporation_data.density_liquid +
                                     (1. - ls.get_value(q_index)) / evaporation_data.density_gas);

              // The normal vector field is oriented such that the normal vector points from
              // the negative level set value (= default for representing the gas phase) to the
              // positive value (= default for representing the liquid phase). Thus, in case the
              // gas phase corresponds to a level set value of 1, the sign of the normal vector
              // has to be changed.
              if (evaporation_data.ls_value_gas == 1.0)
                evapor_vel[q_index] *= -1.0;
            }
        }
      level_set_as_heaviside.zero_out_ghosts();
      normal_vector.zero_out_ghosts();
      evaporative_mass_flux.zero_out_ghosts();

      scratch_data->initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);

      /**
       * write interface velocity to dof vector
       */
      UtilityFunctions::fill_dof_vector_from_cell_operation_vec<dim, dim>(
        evaporation_velocity,
        scratch_data->get_matrix_free(),
        evapor_vel_dof_idx,
        ls_quad_idx,
        scratch_data->get_degree(evapor_vel_dof_idx),           // fe_degree of the resulting vector
        scratch_data->get_degree(ls_hanging_nodes_dof_idx) + 1, // n_q_points_1d of cell operation
        [&](const unsigned int cell,
            const unsigned int quad) -> const Tensor<1, dim, VectorizedArray<double>> & {
          return begin_evaporation_velocity(cell)[quad];
        });

      scratch_data->get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);

      scratch_data->get_pcout(1) << "    | evapor: |u|2 = "
                                 << VectorTools::compute_L2_norm<dim>(evaporation_velocity,
                                                                      *scratch_data,
                                                                      evapor_vel_dof_idx,
                                                                      ls_quad_idx)
                                 << std::endl;

      evaporation_velocity.zero_out_ghosts();
    }

    void
    compute_mass_balance_source_term(VectorType &       mass_balance_rhs,
                                     const unsigned int pressure_dof_idx,
                                     const unsigned int pressure_quad_idx,
                                     bool               zero_out)
    {
      evaporative_mass_flux.update_ghost_values();

      double mass = 0.0;

      scratch_data->get_matrix_free().template cell_loop<VectorType, VectorType>(
        [&](const auto &matrix_free,
            auto &      mass_balance_rhs,
            const auto &level_set_as_heaviside,
            auto        macro_cells) {
          FECellIntegrator<dim, 1, double> heaviside(matrix_free,
                                                     ls_hanging_nodes_dof_idx,
                                                     pressure_quad_idx);

          FECellIntegrator<dim, 1, double> mass_flux(matrix_free,
                                                     pressure_dof_idx,
                                                     pressure_quad_idx);

          FECellIntegrator<dim, 1, double> evap_flux(
            scratch_data->get_matrix_free(),
            ls_hanging_nodes_dof_idx, // @todo: generalize --> temp_dof_idx
            pressure_quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              heaviside.reinit(cell);
              heaviside.read_dof_values_plain(level_set_as_heaviside);
              heaviside.evaluate(false, true);

              mass_flux.reinit(cell);

              evap_flux.reinit(cell);
              evap_flux.read_dof_values_plain(evaporative_mass_flux);
              evap_flux.evaluate(true, false);

              for (unsigned int q_index = 0; q_index < mass_flux.n_q_points; ++q_index)
                {
                  mass_flux.submit_value((1. / evaporation_data.density_liquid -
                                          1. / evaporation_data.density_gas) *
                                           heaviside.get_gradient(q_index).norm() *
                                           evap_flux.get_value(q_index),
                                         q_index);
                  // compute overall rhs
                  for (unsigned int v = 0;
                       v < scratch_data->get_matrix_free().n_active_entries_per_cell_batch(cell);
                       ++v)
                    {
                      mass +=
                        (1. / evaporation_data.density_liquid - 1. / evaporation_data.density_gas) *
                        heaviside.get_gradient(q_index).norm()[v] *
                        evap_flux.get_value(q_index)[v] * mass_flux.JxW(q_index)[v];
                    }
                }

              mass_flux.integrate_scatter(true, false, mass_balance_rhs);
            }
        },
        mass_balance_rhs,
        level_set_as_heaviside,
        zero_out);
      evaporative_mass_flux.zero_out_ghosts();

      scratch_data->get_pcout() << "    | evaporation: jump in the velocity field = "
                                << Utilities::MPI::sum(mass, scratch_data->get_mpi_comm())
                                << std::endl;

      scratch_data->get_pcout() << "    | evapor: |m|2 = " << mass_balance_rhs.l2_norm()
                                << std::endl;
    }

    /*
     * attach functions
     */
    void
    attach_dim_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      evaporation_velocity.update_ghost_values();
      vectors.push_back(&evaporation_velocity);
    }

    void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      evaporative_mass_flux.update_ghost_values();
      vectors.push_back(&evaporative_mass_flux);
    }

    void
    distribute_constraints()
    {
      scratch_data->get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);
      scratch_data->get_constraint(ls_hanging_nodes_dof_idx).distribute(evaporative_mass_flux);
    }

    void
    attach_output_vectors(DataOut<dim> &data_out) const
    {
      /*
       *  evaporation velocity
       */
      evaporation_velocity.update_ghost_values();
      evaporative_mass_flux.update_ghost_values();

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(scratch_data->get_dof_handler(evapor_vel_dof_idx),
                               evaporation_velocity,
                               std::vector<std::string>(dim, "evaporation_velocity"),
                               vector_component_interpretation);
      /*
       *  evaporation mass flux
       */
      data_out.add_data_vector(scratch_data->get_dof_handler(ls_hanging_nodes_dof_idx),
                               evaporative_mass_flux,
                               "evaporative_mass_flux");
    }

    /*
     * getter functions
     */
    inline Tensor<1, dim, VectorizedArray<double>> *
    begin_evaporation_velocity(const unsigned int macro_cell)
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      return &evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                     macro_cell];
    }

    inline const Tensor<1, dim, VectorizedArray<double>> &
    begin_evaporation_velocity(const unsigned int macro_cell) const
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      return evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                    macro_cell];
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const
    {
      return evaporation_velocity;
    }

    LinearAlgebra::distributed::Vector<double> &
    get_velocity()
    {
      return evaporation_velocity;
    }

    const VectorType &
    get_evaporative_mass_flux() const
    {
      return evaporative_mass_flux;
    }

    VectorType &
    get_evaporative_mass_flux()
    {
      return evaporative_mass_flux;
    }

  private:
    /**
     * @todo
     * !!!!!!!! HARD CODED PARAMETERS !!!!!!!!!!!!! --> this function will be replaced when the heat
     * equation is implemented anyhow
     */
    inline double
    compute_temperature_dependent_mass_flux_rate(const double &T,
                                                 const double &pressure_constant,
                                                 const double &temperature_constant,
                                                 const double &boiling_temperature)
    {
      // according to Meier 2020
      const double cs = 1.0;  // sticking coefficent
      const double Cm = 1e-3; // molar_mass/(2*pi*molar_gas_constant)
      return (T >= boiling_temperature) ?
               evaporation_data.evaporative_mass_flux_scale_factor * 0.82 * cs *
                 MeltPool::RecoilPressureOperation<dim>::compute_recoil_pressure_coefficient(
                   T, pressure_constant, temperature_constant, boiling_temperature) *
                 std::sqrt(Cm / T) :
               0.0;
    }
  };
} // namespace MeltPoolDG::Evaporation
