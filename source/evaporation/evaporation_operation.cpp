#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_continuous.hpp>
#include <meltpooldg/evaporation/evaporation_mass_flux_operator_interface_value.hpp>
#include <meltpooldg/evaporation/evaporation_model_hardt_wondra.hpp>
#include <meltpooldg/evaporation/evaporation_model_recoil_pressure.hpp>
#include <meltpooldg/evaporation/evaporation_operation.hpp>
#include <meltpooldg/melt_pool/recoil_pressure_operation.hpp>
#include <meltpooldg/utilities/physical_constants.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  template <int dim>
  EvaporationOperation<dim>::EvaporationOperation(
    const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
    const VectorType &                             level_set_as_heaviside_in,
    const BlockVectorType &                        normal_vector_in,
    std::shared_ptr<SimulationBase<dim>>           base_in,
    const unsigned int                             normal_dof_idx_in,
    const unsigned int                             evapor_vel_dof_idx_in,
    const unsigned int                             ls_hanging_nodes_dof_idx_in,
    const unsigned int                             ls_quad_idx_in,
    const VectorType *                             temperature,
    const unsigned int                             temp_dof_idx)

    : scratch_data(scratch_data_in)
    , evaporation_data(base_in->parameters.evapor)
    , material(base_in->parameters.material)
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
    , temperature(temperature)
    , temp_dof_idx(temp_dof_idx)
  {
    AssertThrow(material.first.density > 0.0 && material.second.density > 0.0,
                ExcMessage("The materials' densities must be greater than zero! Abort..."));
    reinit();
  }

  template <int dim>
  void
  EvaporationOperation<dim>::register_temperature_vector(const VectorType * temperature,
                                                         const unsigned int temp_dof_idx_)
  {
    temperature  = temperature;
    temp_dof_idx = temp_dof_idx_;
  }

  //@todo move to constructor
  template <int dim>
  void
  EvaporationOperation<dim>::register_evaporative_mass_flux_model(
    const RecoilPressureData<double> &recoil_data,
    const VectorType &                distance)
  {
    /*                 .
     * local operation m(T)
     */
    //@todo: add asserts of parameters
    if (evaporation_data.evaporation_model == "constant")
      { /* do nothing --> no model has to be set up */
      }
    else if (evaporation_data.evaporation_model == "recoil pressure")
      evapor_model = std::make_shared<EvaporationModelRecoilPressure<dim>>(
        evaporation_data.boiling_temperature,
        recoil_data.pressure_constant,
        recoil_data.temperature_constant,
        evaporation_data.evaporative_mass_flux_scale_factor);
    else if (evaporation_data.evaporation_model == "Hardt Wondra")
      evapor_model =
        std::make_shared<EvaporationModelHardtWondra>(evaporation_data.coefficient,
                                                      evaporation_data.latent_heat_of_evaporation,
                                                      material.first.density,
                                                      evaporation_data.molar_mass,
                                                      evaporation_data.boiling_temperature);
    else
      AssertThrow(false, ExcNotImplemented());
    /*
     * Computation of DoF-Vector
     */
    if (evaporation_data.formulation_evaporative_mass_flux_over_interface == "continuous")
      evapor_mass_flux_operator =
        std::make_shared<EvaporationMassFluxOperatorContinuous<dim>>(*scratch_data,
                                                                     *evapor_model,
                                                                     temp_dof_idx);
    else if (evaporation_data.formulation_evaporative_mass_flux_over_interface == "interface value")
      {
        evapor_mass_flux_operator =
          std::make_shared<EvaporationMassFluxOperatorInterfaceValue<dim>>(*scratch_data,
                                                                           *evapor_model,
                                                                           level_set_as_heaviside,
                                                                           distance,
                                                                           normal_vector,
                                                                           ls_hanging_nodes_dof_idx,
                                                                           temp_dof_idx);
      }
    else if (evaporation_data.formulation_evaporative_mass_flux_over_interface == "line integral")
      AssertThrow(false, ExcNotImplemented());
  }

  template <int dim>
  void
  EvaporationOperation<dim>::compute_evaporative_mass_flux()
  {
    if (evaporation_data.evaporation_model == "constant")
      evaporative_mass_flux = evaporation_data.evaporative_mass_flux;
    else
      evapor_mass_flux_operator->compute_evaporative_mass_flux(evaporative_mass_flux, *temperature);
  }

  template <int dim>
  void
  EvaporationOperation<dim>::compute_evaporation_velocity(
    const std::string &interpolation_type_parameters)
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

    evaporation_velocities.resize(scratch_data->get_matrix_free().n_cell_batches() * ls.n_q_points);

    for (unsigned int cell = 0; cell < scratch_data->get_matrix_free().n_cell_batches(); ++cell)
      {
        Tensor<1, dim, VectorizedArray<double>> *evapor_vel = begin_evaporation_velocity(cell);

        ls.reinit(cell);
        ls.read_dof_values(level_set_as_heaviside);
        ls.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        normal_vec.reinit(cell);
        normal_vec.read_dof_values(normal_vector);
        normal_vec.evaluate(EvaluationFlags::values);

        evap_flux.reinit(cell);
        evap_flux.read_dof_values(evaporative_mass_flux);
        evap_flux.evaluate(EvaluationFlags::values);

        for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
          {
            const auto n_phi =
              MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index),
                                                      tolerance_normal_vector);

            //              ρ
            // evaluate  ------
            //            dρ/dΦ
            VectorizedArray<double> rho_d_rho_d_phi = 1.0;

            if (interpolation_type_parameters == "true")
              {
                // clang-format off
                  rho_d_rho_d_phi = (ls.get_value(q_index) * material.second.density + (1.-ls.get_value(q_index) * material.first.density)) 
                    / //-----------------------------------------------------------------------------------------------------
                                                    (material.second.density - material.first.density);
                // clang-format on
              }

            evapor_vel[q_index] = n_phi * evap_flux.get_value(q_index) * rho_d_rho_d_phi *
                                  UtilityFunctions::interpolate(ls.get_value(q_index),
                                                                1. / material.first.density,
                                                                1. / material.second.density);

            // The normal vector field is oriented such that the normal vector points from
            // the negative level set value (= default for representing the gas phase) to the
            // positive value (= default for representing the liquid phase). Thus, in case the
            // gas phase corresponds to a level set value of 1, the sign of the normal vector
            // has to be changed.
            if (evaporation_data.ls_value_gas == 1.0)
              AssertThrow(false, ExcNotImplemented());
            // evapor_vel[q_index] *= -1.0;
          }
      }
    level_set_as_heaviside.zero_out_ghost_values();
    normal_vector.zero_out_ghost_values();
    evaporative_mass_flux.zero_out_ghost_values();

    scratch_data->initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);

    /**
     * write interface velocity to dof vector
     */
    if (scratch_data->is_hex_mesh())
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

    evaporation_velocity.zero_out_ghost_values();
  }

  template <int dim>
  void
  EvaporationOperation<dim>::compute_mass_balance_source_term(VectorType &       mass_balance_rhs,
                                                              const unsigned int pressure_dof_idx,
                                                              const unsigned int pressure_quad_idx,
                                                              bool               zero_out)
  {
    evaporative_mass_flux.update_ghost_values();
    if (temperature)
      temperature->update_ghost_values();


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
            heaviside.evaluate(EvaluationFlags::gradients);

            mass_flux.reinit(cell);

            evap_flux.reinit(cell);
            evap_flux.read_dof_values_plain(evaporative_mass_flux);
            evap_flux.evaluate(EvaluationFlags::values);

            for (unsigned int q_index = 0; q_index < mass_flux.n_q_points; ++q_index)
              {
                mass_flux.submit_value((1. / material.second.density -
                                        1. / material.first.density) *
                                         heaviside.get_gradient(q_index).norm() *
                                         evap_flux.get_value(q_index),
                                       q_index);
                // compute overall rhs
                for (unsigned int v = 0;
                     v < scratch_data->get_matrix_free().n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    mass += (1. / material.second.density - 1. / material.first.density) *
                            heaviside.get_gradient(q_index).norm()[v] *
                            evap_flux.get_value(q_index)[v] * mass_flux.JxW(q_index)[v];
                  }
              }

            mass_flux.integrate_scatter(EvaluationFlags::values, mass_balance_rhs);
          }
      },
      mass_balance_rhs,
      level_set_as_heaviside,
      zero_out);
    evaporative_mass_flux.zero_out_ghost_values();

    scratch_data->get_pcout() << "    | evaporation: jump in the velocity field = "
                              << Utilities::MPI::sum(mass, scratch_data->get_mpi_comm())
                              << std::endl;

    scratch_data->get_pcout() << "    | evapor: |m|2 = " << mass_balance_rhs.l2_norm() << std::endl;
  }

  template <int dim>
  void
  EvaporationOperation<dim>::reinit()
  {
    scratch_data->initialize_dof_vector(
      evaporative_mass_flux, ls_hanging_nodes_dof_idx); // @todo: evapor_dof_idx/temp_dof_idx
    scratch_data->initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);
  }

  template <int dim>
  void
  EvaporationOperation<dim>::attach_dim_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    evaporation_velocity.update_ghost_values();
    vectors.push_back(&evaporation_velocity);
  }

  template <int dim>
  void
  EvaporationOperation<dim>::attach_vectors(
    std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
  {
    evaporative_mass_flux.update_ghost_values();
    vectors.push_back(&evaporative_mass_flux);
  }

  template <int dim>
  void
  EvaporationOperation<dim>::distribute_constraints()
  {
    scratch_data->get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);
    scratch_data->get_constraint(ls_hanging_nodes_dof_idx).distribute(evaporative_mass_flux);
  }

  template <int dim>
  void
  EvaporationOperation<dim>::attach_output_vectors(GenericDataOut<dim> &data_out) const
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

  template <int dim>
  inline Tensor<1, dim, VectorizedArray<double>> *
  EvaporationOperation<dim>::begin_evaporation_velocity(const unsigned int macro_cell)
  {
    AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
    return &evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                   macro_cell];
  }

  template <int dim>
  inline const Tensor<1, dim, VectorizedArray<double>> &
  EvaporationOperation<dim>::begin_evaporation_velocity(const unsigned int macro_cell) const
  {
    AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
    return evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                  macro_cell];
  }

  template <int dim>
  const LinearAlgebra::distributed::Vector<double> &
  EvaporationOperation<dim>::get_velocity() const
  {
    return evaporation_velocity;
  }

  template <int dim>
  LinearAlgebra::distributed::Vector<double> &
  EvaporationOperation<dim>::get_velocity()
  {
    return evaporation_velocity;
  }

  template <int dim>
  const VectorType &
  EvaporationOperation<dim>::get_evaporative_mass_flux() const
  {
    return evaporative_mass_flux;
  }

  template <int dim>
  VectorType &
  EvaporationOperation<dim>::get_evaporative_mass_flux()
  {
    return evaporative_mass_flux;
  }

  template class EvaporationOperation<1>;
  template class EvaporationOperation<2>;
  template class EvaporationOperation<3>;
} // namespace MeltPoolDG::Evaporation
