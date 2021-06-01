#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_continuous.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim>
  EvaporationMassFluxOperatorContinuous<dim>::EvaporationMassFluxOperatorContinuous(
    const ScratchData<dim> &    scratch_data,
    const EvaporationModelBase &evaporation_model,
    const unsigned int          temp_dof_idx)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
    , temp_dof_idx(temp_dof_idx)
  {}


  template <int dim>
  void
  EvaporationMassFluxOperatorContinuous<dim>::compute_evaporative_mass_flux(
    VectorType &      evaporative_mass_flux,
    const VectorType &temperature) const
  {
    temperature.update_ghost_values();
    const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(temp_dof_idx);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : scratch_data.get_dof_handler(temp_dof_idx).active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            evaporative_mass_flux[local_dof_indices[i]] =
              evaporation_model.local_compute_evaporative_mass_flux(
                temperature[local_dof_indices[i]]);
        }
    temperature.zero_out_ghost_values();
  }

  template class EvaporationMassFluxOperatorContinuous<1>;
  template class EvaporationMassFluxOperatorContinuous<2>;
  template class EvaporationMassFluxOperatorContinuous<3>;
} // namespace MeltPoolDG::Evaporation
