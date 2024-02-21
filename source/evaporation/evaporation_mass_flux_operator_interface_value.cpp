#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_interface_value.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim>
  EvaporationMassFluxOperatorInterfaceValue<dim>::EvaporationMassFluxOperatorInterfaceValue(
    const ScratchData<dim>                   &scratch_data,
    const LevelSet::NearestPointData<double> &data,
    const EvaporationModelBase               &evaporation_model,
    const VectorType                         &level_set_as_heaviside,
    const VectorType                         &distance,
    const BlockVectorType                    &normal_vector,
    const unsigned int                        ls_dof_idx_in,
    const unsigned int                        temp_hanging_nodes_dof_idx_in,
    const unsigned int                        evapor_mass_flux_dof_idx_in)
    : scratch_data(scratch_data)
    , nearest_point_data(data)
    , evaporation_model(evaporation_model)
    , level_set_as_heaviside(level_set_as_heaviside)
    , distance(distance)
    , normal_vector(normal_vector)
    , ls_dof_idx(ls_dof_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , evapor_mass_flux_dof_idx(evapor_mass_flux_dof_idx_in)
    , tolerance_normal_vector(
        UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                              scratch_data.get_mapping()))
  {}


  template <int dim>
  void
  EvaporationMassFluxOperatorInterfaceValue<dim>::compute_evaporative_mass_flux(
    VectorType       &evaporative_mass_flux,
    const VectorType &temperature) const
  {
    if (!nearest_point_search)
      nearest_point_search = std::make_unique<LevelSet::Tools::NearestPoint<dim>>(
        scratch_data.get_mapping(),
        scratch_data.get_dof_handler(ls_dof_idx),
        distance,
        normal_vector,
        scratch_data.get_remote_point_evaluation(evapor_mass_flux_dof_idx),
        nearest_point_data,
        scratch_data.get_timer());

    nearest_point_search->reinit(scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx));

    // get temperature values at projected interface points
    scratch_data.initialize_dof_vector(evaporative_mass_flux, evapor_mass_flux_dof_idx);
    evaporative_mass_flux = 0.0;

    nearest_point_search->fill_dof_vector_with_point_values(
      evaporative_mass_flux,
      scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
      temperature,
      true /*zero out*/,
      [&](const double x) { return evaporation_model.local_compute_evaporative_mass_flux(x); });

    scratch_data.get_constraint(evapor_mass_flux_dof_idx).distribute(evaporative_mass_flux);
  }

  template class EvaporationMassFluxOperatorInterfaceValue<1>;
  template class EvaporationMassFluxOperatorInterfaceValue<2>;
  template class EvaporationMassFluxOperatorInterfaceValue<3>;
} // namespace MeltPoolDG::Evaporation
