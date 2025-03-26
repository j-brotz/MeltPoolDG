#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_interface_value.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim, typename number>
  EvaporationMassFluxOperatorInterfaceValue<dim, number>::EvaporationMassFluxOperatorInterfaceValue(
    const ScratchData<dim, dim, number>      &scratch_data,
    const LevelSet::NearestPointData<number> &data,
    const EvaporationModelBase<number>       &evaporation_model,
    const VectorType                         &level_set_as_heaviside,
    const VectorType                         &distance,
    const BlockVectorType                    &normal_vector,
    const unsigned int                        ls_dof_idx_in,
    const unsigned int                        heat_hanging_nodes_dof_idx_in,
    const unsigned int                        evapor_mass_flux_dof_idx_in)
    : scratch_data(scratch_data)
    , nearest_point_data(data)
    , evaporation_model(evaporation_model)
    , level_set_as_heaviside(level_set_as_heaviside)
    , distance(distance)
    , normal_vector(normal_vector)
    , ls_dof_idx(ls_dof_idx_in)
    , heat_hanging_nodes_dof_idx(heat_hanging_nodes_dof_idx_in)
    , evapor_mass_flux_dof_idx(evapor_mass_flux_dof_idx_in)
    , tolerance_normal_vector(
        UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                              scratch_data.get_mapping()))
  {}


  template <int dim, typename number>
  void
  EvaporationMassFluxOperatorInterfaceValue<dim, number>::compute_evaporative_mass_flux(
    VectorType       &evaporative_mass_flux,
    const VectorType &temperature) const
  {
    if (not nearest_point_search)
      nearest_point_search = std::make_unique<LevelSet::Tools::NearestPoint<dim, double>>(
        scratch_data.get_mapping(),
        scratch_data.get_dof_handler(ls_dof_idx),
        distance,
        normal_vector,
        scratch_data.get_remote_point_evaluation(evapor_mass_flux_dof_idx),
        nearest_point_data,
        scratch_data.get_timer());

    nearest_point_search->reinit(&scratch_data.get_dof_handler(heat_hanging_nodes_dof_idx));

    // get temperature values at projected interface points
    scratch_data.initialize_dof_vector(evaporative_mass_flux, evapor_mass_flux_dof_idx);
    evaporative_mass_flux = 0.0;

    nearest_point_search->fill_dof_vector_with_point_values(
      evaporative_mass_flux, temperature, true /*zero out*/, [&](const number x) {
        return evaporation_model.local_compute_evaporative_mass_flux(x);
      });

    scratch_data.get_constraint(evapor_mass_flux_dof_idx).distribute(evaporative_mass_flux);
  }

  template class EvaporationMassFluxOperatorInterfaceValue<1, double>;
  template class EvaporationMassFluxOperatorInterfaceValue<2, double>;
  template class EvaporationMassFluxOperatorInterfaceValue<3, double>;
} // namespace MeltPoolDG::Evaporation
