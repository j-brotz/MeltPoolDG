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
    const ScratchData<dim> &    scratch_data,
    const EvaporationModelBase &evaporation_model,
    const VectorType &          level_set_as_heaviside,
    const VectorType &          distance,
    const BlockVectorType &     normal_vector,
    const unsigned int          ls_dof_idx_in,
    const unsigned int          temp_hanging_nodes_dof_idx_in,
    const unsigned int          evapor_mass_flux_dof_idx_in,
    const unsigned int          n_iterations)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
    , level_set_as_heaviside(level_set_as_heaviside)
    , distance(distance)
    , normal_vector(normal_vector)
    , ls_dof_idx(ls_dof_idx_in)
    , temp_hanging_nodes_dof_idx(temp_hanging_nodes_dof_idx_in)
    , evapor_mass_flux_dof_idx(evapor_mass_flux_dof_idx_in)
    , n_iterations(n_iterations)
    , tolerance_normal_vector(
        UtilityFunctions::compute_numerical_zero_of_norm<dim>(scratch_data.get_triangulation(),
                                                              scratch_data.get_mapping()))
  {
    AssertThrow(scratch_data.get_degree(temp_hanging_nodes_dof_idx) ==
                  scratch_data.get_degree(ls_dof_idx),
                ExcMessage("This algorithm is currently only supported for the same degree "
                           "between the heat transfer and the level set."));
  }


  template <int dim>
  void
  EvaporationMassFluxOperatorInterfaceValue<dim>::compute_evaporative_mass_flux(
    VectorType &      evaporative_mass_flux,
    const VectorType &temperature) const
  {
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation(
      1e-6 /*tolerance*/, true /*unique mapping*/);

    const auto [dof_indices, evaluation_points] =
      LevelSet::Tools::compute_projected_points_at_interface<dim>(
        scratch_data.get_mapping(),
        scratch_data.get_dof_handler(ls_dof_idx),
        scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
        level_set_as_heaviside,
        distance,
        normal_vector,
        remote_point_evaluation,
        n_iterations);
    /*
     * get temperature values at projected interface points
     */

    remote_point_evaluation.reinit(evaluation_points,
                                   scratch_data.get_triangulation(),
                                   scratch_data.get_mapping());

    temperature.update_ghost_values();

    const auto temperature_evaluation_values =
      dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                           scratch_data.get_dof_handler(temp_hanging_nodes_dof_idx),
                                           temperature);
    temperature.zero_out_ghost_values();

    Assert(temperature_evaluation_values.size() == evaluation_points.size(),
           ExcMessage("The size of vectors must match."));

    /*
     * compute evaporative mass flux from the temperature value at the interface
     */
    scratch_data.initialize_dof_vector(evaporative_mass_flux, evapor_mass_flux_dof_idx);
    evaporative_mass_flux = 0.0;

    for (unsigned int i = 0; i < evaluation_points.size(); ++i)
      {
        evaporative_mass_flux[dof_indices[i]] =
          evaporation_model.local_compute_evaporative_mass_flux(temperature_evaluation_values[i]);
      }

    evaporative_mass_flux.update_ghost_values(); //@todo: zero out ghost values

    distance.zero_out_ghost_values();
    normal_vector.zero_out_ghost_values();
    level_set_as_heaviside.zero_out_ghost_values();
  }

  template class EvaporationMassFluxOperatorInterfaceValue<1>;
  template class EvaporationMassFluxOperatorInterfaceValue<2>;
  template class EvaporationMassFluxOperatorInterfaceValue<3>;
} // namespace MeltPoolDG::Evaporation
