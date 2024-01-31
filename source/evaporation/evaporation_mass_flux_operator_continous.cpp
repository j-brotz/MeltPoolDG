#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/evaporation/evaporation_mass_flux_operator_continuous.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim>
  EvaporationMassFluxOperatorContinuous<dim>::EvaporationMassFluxOperatorContinuous(
    const ScratchData<dim>     &scratch_data,
    const EvaporationModelBase &evaporation_model)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
  {}


  template <int dim>
  void
  EvaporationMassFluxOperatorContinuous<dim>::compute_evaporative_mass_flux(
    VectorType       &evaporative_mass_flux,
    const VectorType &temperature) const
  {
    const bool update_ghosts = !temperature.has_ghost_elements();
    if (update_ghosts)
      temperature.update_ghost_values();

    for (unsigned int i = 0; i < evaporative_mass_flux.locally_owned_size(); ++i)
      evaporative_mass_flux.local_element(i) =
        evaporation_model.local_compute_evaporative_mass_flux(temperature.local_element(i));

    if (update_ghosts)
      temperature.zero_out_ghost_values();
  }

  template class EvaporationMassFluxOperatorContinuous<1>;
  template class EvaporationMassFluxOperatorContinuous<2>;
  template class EvaporationMassFluxOperatorContinuous<3>;
} // namespace MeltPoolDG::Evaporation
