#pragma once

#include <meltpooldg/phase_change/evaporation_data.hpp>
#include <meltpooldg/phase_change/evaporation_model_base.hpp>
#include <meltpooldg/phase_change/recoil_pressure_data.hpp>
#include <meltpooldg/utilities/material_data.hpp>

#include <memory>


namespace MeltPoolDG::Evaporation
{
  std::unique_ptr<EvaporationModelBase>
  get_evaporation_model(const EvaporationData<double> &evapor_data,
                        const MaterialData<double>    &material_data);

} // namespace MeltPoolDG::Evaporation