#include <meltpooldg/phase_change/evaporation_operation.hpp>
//
#include <deal.II/base/exceptions.h>

#include <deal.II/numerics/data_component_interpretation.h>

#include <meltpooldg/phase_change/evaporation_mass_flux_operator_continuous.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_interface_value.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_thickness_integration.hpp>
#include <meltpooldg/phase_change/evaporation_model_constant.hpp>
#include <meltpooldg/phase_change/evaporation_model_factory.hpp>
#include <meltpooldg/phase_change/evaporation_source_terms_continuous.hpp>
#include <meltpooldg/phase_change/evaporation_source_terms_sharp.hpp>
#include <meltpooldg/utilities/journal.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

#include <string>


namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim, typename number>
  EvaporationOperation<dim, number>::EvaporationOperation(
    const ScratchData<dim, dim, number> &scratch_data_in,
    const VectorType                    &level_set_as_heaviside_in,
    const BlockVectorType               &normal_vector_in,
    const EvaporationData<number>       &evapor_data_in,
    const MaterialData<number>          &material_data_in,
    const unsigned int                   normal_dof_idx_in,
    const unsigned int                   evapor_vel_dof_idx_in,
    const unsigned int                   evapor_vel_quad_idx_in,
    const unsigned int                   evapor_mass_flux_dof_idx_in,
    const unsigned int                   ls_hanging_nodes_dof_idx_in,
    const unsigned int                   ls_quad_idx_in)

    : scratch_data(scratch_data_in)
    , evapor_data(evapor_data_in)
    , material_data(material_data_in)
    , level_set_as_heaviside(level_set_as_heaviside_in)
    , normal_vector(normal_vector_in)
    , normal_dof_idx(normal_dof_idx_in)
    , evapor_vel_dof_idx(evapor_vel_dof_idx_in)
    , evapor_mass_flux_dof_idx(evapor_mass_flux_dof_idx_in)
    , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx_in)
    , ls_quad_idx(ls_quad_idx_in)
    , tolerance_normal_vector(UtilityFunctions::compute_numerical_zero_of_norm<dim, number>(
        scratch_data.get_triangulation(),
        scratch_data.get_mapping()))
  {
    AssertThrow(material_data.gas.density > 0.0 && material_data.liquid.density > 0.0,
                ExcMessage("The materials' densities must be greater than zero! Abort..."));

    if (evapor_data.evaporative_dilation_rate.model == InterfaceFluxType::regularized)
      evapor_source_terms_operator =
        std::make_shared<EvaporationSourceTermsContinuous<dim, number>>(
          scratch_data,
          evapor_data,
          level_set_as_heaviside,
          normal_vector,
          evaporative_mass_flux,
          ls_hanging_nodes_dof_idx,
          ls_quad_idx,
          normal_dof_idx,
          evapor_vel_dof_idx,
          evapor_vel_quad_idx_in,
          evapor_mass_flux_dof_idx,
          tolerance_normal_vector,
          material_data.gas.density,
          material_data.liquid.density,
          material_data.two_phase_fluid_properties_transition_type);
    else if (evapor_data.evaporative_dilation_rate.model == InterfaceFluxType::sharp)
      evapor_source_terms_operator =
        std::make_shared<EvaporationSourceTermsSharp<dim, number>>(scratch_data,
                                                                   evapor_data,
                                                                   level_set_as_heaviside,
                                                                   normal_vector,
                                                                   evaporative_mass_flux,
                                                                   ls_hanging_nodes_dof_idx,
                                                                   ls_quad_idx,
                                                                   normal_dof_idx,
                                                                   evapor_vel_dof_idx,
                                                                   evapor_vel_quad_idx_in,
                                                                   evapor_mass_flux_dof_idx,
                                                                   tolerance_normal_vector,
                                                                   material_data.gas.density,
                                                                   material_data.liquid.density);
    else
      Assert(false, ExcMessage("Specified evaporation source term is not implemented!"));

    if (evapor_data.evaporative_mass_flux_model == EvaporationModelType::analytical)
      {
        evapor_model =
          std::make_shared<EvaporationModelConstant<number>>(evapor_data.analytical.function);
      }
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::register_surface_mesh(
    const std::vector<std::tuple<const typename Triangulation<dim, dim>::cell_iterator /*cell*/,
                                 std::vector<Point<dim>> /*quad_points*/,
                                 std::vector<number> /*weights*/
                                 >> &surface_mesh_info_in)
  {
    dynamic_cast<EvaporationSourceTermsSharp<dim, number> *>(evapor_source_terms_operator.get())
      ->register_surface_mesh(surface_mesh_info_in);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::reinit(
    const VectorType                             *temperature_in,
    const VectorType                             &distance,
    const LevelSet::NearestPointData<number>     &nearest_point_data,
    const LevelSet::ReinitializationData<number> &reinit_data,
    const unsigned int                            heat_dof_idx_in)
  {
    temperature  = temperature_in;
    heat_dof_idx = heat_dof_idx_in;

    // setup of temperature-dependent evaporation models
    evapor_model = get_evaporation_model(evapor_data, material_data);

    /*
     * Computation of DoF-Vector
     */
    if (evapor_data.interface_temperature_evaluation_type ==
        EvaporativeMassFluxTemperatureEvaluationType::local_value)
      evapor_mass_flux_operator =
        std::make_shared<EvaporationMassFluxOperatorContinuous<dim, number>>(scratch_data,
                                                                             *evapor_model);
    else if (evapor_data.interface_temperature_evaluation_type ==
             EvaporativeMassFluxTemperatureEvaluationType::interface_value)
      {
        evapor_mass_flux_operator =
          std::make_shared<EvaporationMassFluxOperatorInterfaceValue<dim, number>>(
            scratch_data,
            nearest_point_data,
            *evapor_model,
            level_set_as_heaviside,
            distance,
            normal_vector,
            ls_hanging_nodes_dof_idx,
            heat_dof_idx,
            evapor_mass_flux_dof_idx);
      }
    else if (evapor_data.interface_temperature_evaluation_type ==
             EvaporativeMassFluxTemperatureEvaluationType::thickness_integral)
      evapor_mass_flux_operator =
        std::make_shared<EvaporationMassFluxOperatorThicknessIntegration<dim, number>>(
          scratch_data,
          *evapor_model,
          evapor_data.thickness_integral,
          reinit_data,
          level_set_as_heaviside,
          normal_vector,
          ls_hanging_nodes_dof_idx,
          normal_dof_idx,
          heat_dof_idx,
          evapor_mass_flux_dof_idx);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::set_time(const number &time_in)
  {
    time = time_in;
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::compute_evaporative_mass_flux()
  {
    // prescribe spatially constant, potentially time-dependent evaporative mass flux
    if (evapor_data.evaporative_mass_flux_model == EvaporationModelType::analytical)
      {
        evaporative_mass_flux = evapor_model->local_compute_evaporative_mass_flux(time);
      }
    else
      {
        Assert(evapor_mass_flux_operator && temperature,
               ExcMessage(
                 "Before computing the evaporative mass flux, register_evaporative_mass_flux_model "
                 "needs to be called."));
        evapor_mass_flux_operator->compute_evaporative_mass_flux(evaporative_mass_flux,
                                                                 *temperature);
      }

    Journal::print_formatted_norm<number>(
      scratch_data.get_pcout(2),
      [&]() -> number {
        return MeltPoolDG::VectorTools::compute_norm<dim, number>(evaporative_mass_flux,
                                                                  scratch_data,
                                                                  evapor_mass_flux_dof_idx,
                                                                  ls_quad_idx);
      },
      "evaporative_mass_flux",
      "evaporation_operation",
      10);

    // update ghost values of current solution
    evaporative_mass_flux.update_ghost_values();
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::compute_evaporation_velocity()
  {
    evapor_source_terms_operator->compute_evaporation_velocity(evaporation_velocity);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::compute_level_set_source_term(
    VectorType        &ls_rhs,
    const unsigned int ls_dof_idx,
    const VectorType  &level_set,
    const unsigned int pressure_dof_idx)
  {
    evapor_source_terms_operator->compute_level_set_source_term(ls_rhs,
                                                                ls_dof_idx,
                                                                level_set,
                                                                pressure_dof_idx);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::compute_mass_balance_source_term(
    VectorType        &mass_balance_rhs,
    const unsigned int pressure_dof_idx,
    const unsigned int pressure_quad_idx,
    bool               zero_out)
  {
    if (zero_out)
      scratch_data.initialize_dof_vector(mass_balance_rhs, pressure_dof_idx);

    evapor_source_terms_operator->compute_mass_balance_source_term(mass_balance_rhs,
                                                                   pressure_dof_idx,
                                                                   pressure_quad_idx,
                                                                   zero_out);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::reinit()
  {
    scratch_data.initialize_dof_vector(evaporative_mass_flux, evapor_mass_flux_dof_idx);
    scratch_data.initialize_dof_vector(evaporation_velocity, evapor_vel_dof_idx);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::attach_dim_vectors(std::vector<VectorType *> &vectors)
  {
    // TODO: delete -- not needed
    vectors.push_back(&evaporation_velocity);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::attach_vectors(std::vector<VectorType *> &vectors)
  {
    vectors.push_back(&evaporative_mass_flux);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::distribute_constraints()
  {
    scratch_data.get_constraint(evapor_vel_dof_idx).distribute(evaporation_velocity);
    scratch_data.get_constraint(evapor_mass_flux_dof_idx).distribute(evaporative_mass_flux);
  }

  template <int dim, typename number>
  void
  EvaporationOperation<dim, number>::attach_output_vectors(
    GenericDataOut<dim, number> &data_out) const
  {
    /*
     *  evaporation velocity
     */
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      vector_component_interpretation(dim,
                                      DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(scratch_data.get_dof_handler(evapor_vel_dof_idx),
                             evaporation_velocity,
                             std::vector<std::string>(dim, "evaporation_velocity"),
                             vector_component_interpretation);
    /*
     *  evaporation mass flux
     */
    data_out.add_data_vector(scratch_data.get_dof_handler(evapor_mass_flux_dof_idx),
                             evaporative_mass_flux,
                             "evaporative_mass_flux");
  }

  template <int dim, typename number>
  inline Tensor<1, dim, VectorizedArray<number>> *
  EvaporationOperation<dim, number>::begin_evaporation_velocity(const unsigned int macro_cell)
  {
    AssertIndexRange(macro_cell, scratch_data.get_matrix_free().n_cell_batches());
    return &evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                   macro_cell];
  }

  template <int dim, typename number>
  inline const Tensor<1, dim, VectorizedArray<number>> &
  EvaporationOperation<dim, number>::begin_evaporation_velocity(const unsigned int macro_cell) const
  {
    AssertIndexRange(macro_cell, scratch_data.get_matrix_free().n_cell_batches());
    return evaporation_velocities[scratch_data.get_matrix_free().get_n_q_points(ls_quad_idx) *
                                  macro_cell];
  }

  template <int dim, typename number>
  const LinearAlgebra::distributed::Vector<number> &
  EvaporationOperation<dim, number>::get_velocity() const
  {
    return evaporation_velocity;
  }

  template <int dim, typename number>
  typename EvaporationOperation<dim, number>::VectorType &
  EvaporationOperation<dim, number>::get_velocity()
  {
    return evaporation_velocity;
  }

  template <int dim, typename number>
  const typename EvaporationOperation<dim, number>::VectorType &
  EvaporationOperation<dim, number>::get_evaporative_mass_flux() const
  {
    return evaporative_mass_flux;
  }

  template <int dim, typename number>
  LinearAlgebra::distributed::Vector<number> &
  EvaporationOperation<dim, number>::get_evaporative_mass_flux()
  {
    return evaporative_mass_flux;
  }

  template class EvaporationOperation<1, double>;
  template class EvaporationOperation<2, double>;
  template class EvaporationOperation<3, double>;
} // namespace MeltPoolDG::Evaporation
