#include <deal.II/lac/la_parallel_vector.h>

#include <meltpooldg/core/scratch_data.hpp>
#include <meltpooldg/phase_change/evaporation_mass_flux_operator_continuous.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;

  template <int dim, typename number>
  EvaporationMassFluxOperatorContinuous<dim, number>::EvaporationMassFluxOperatorContinuous(
    const ScratchData<dim, dim, number> &scratch_data,
    const EvaporationModelBase<number>  &evaporation_model)
    : scratch_data(scratch_data)
    , evaporation_model(evaporation_model)
  {}


  template <int dim, typename number>
  void
  EvaporationMassFluxOperatorContinuous<dim, number>::compute_evaporative_mass_flux(
    VectorType       &evaporative_mass_flux,
    const VectorType &temperature) const
  {
    const bool update_ghosts = !temperature.has_ghost_elements();
    if (update_ghosts)
      temperature.update_ghost_values();

    for (unsigned int i = 0; i < evaporative_mass_flux.locally_owned_size(); ++i)
      {
        // TODO
        if (temperature.local_element(i) == 0)
          evaporative_mass_flux.local_element(i) = 0;
        else
          evaporative_mass_flux.local_element(i) =
            evaporation_model.local_compute_evaporative_mass_flux(temperature.local_element(i));
      }

    if (update_ghosts)
      temperature.zero_out_ghost_values();
  }

  template class EvaporationMassFluxOperatorContinuous<1, double>;
  template class EvaporationMassFluxOperatorContinuous<2, double>;
  template class EvaporationMassFluxOperatorContinuous<3, double>;
} // namespace MeltPoolDG::Evaporation
