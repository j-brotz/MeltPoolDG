#pragma once

#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>
#include <meltpooldg/utilities/material_data.hpp>

namespace MeltPoolDG::Evaporation
{
  template <typename number>
  std::unique_ptr<EvaporationModelBase<number>>
  get_evaporation_model(const EvaporationData<number> &evapor_data,
                        const MaterialData<number>    &material_data);

} // namespace MeltPoolDG::Evaporation
