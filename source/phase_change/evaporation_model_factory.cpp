#include <meltpooldg/phase_change/evaporation_model_factory.hpp>
//
#include <deal.II/base/exceptions.h>

#include <meltpooldg/phase_change/evaporation_model_constant.hpp>
#include <meltpooldg/phase_change/evaporation_model_hardt_wondra.hpp>
#include <meltpooldg/phase_change/evaporation_model_recoil_pressure.hpp>
#include <meltpooldg/phase_change/evaporation_model_saturated_vapor_pressure.hpp>


namespace MeltPoolDG::Evaporation
{
  std::unique_ptr<EvaporationModelBase>
  get_evaporation_model(const EvaporationData<double> &evapor_data,
                        const MaterialData<double>    &material_data)
  {
    std::unique_ptr<EvaporationModelBase> evapor_model;

    switch (evapor_data.evaporative_mass_flux_model)
      {
          case EvaporationModelType::recoil_pressure: {
            evapor_model = std::make_unique<EvaporationModelRecoilPressure>(
              evapor_data.recoil,
              material_data.boiling_temperature,
              material_data.molar_mass,
              material_data.latent_heat_of_evaporation);
            break;
          }
          case EvaporationModelType::saturated_vapor_pressure: {
            evapor_model = std::make_unique<EvaporationModelSaturatedVaporPressure>(
              evapor_data.recoil,
              material_data.boiling_temperature,
              material_data.molar_mass,
              material_data.latent_heat_of_evaporation);
            break;
          }
          case EvaporationModelType::hardt_wondra: {
            evapor_model = std::make_unique<EvaporationModelHardtWondra>(
              evapor_data.hardt_wondra.coefficient,
              material_data.latent_heat_of_evaporation,
              material_data.gas.density,
              material_data.molar_mass,
              material_data.boiling_temperature);
            break;
          }
          case EvaporationModelType::analytical: {
            evapor_model =
              std::make_unique<EvaporationModelConstant>(evapor_data.analytical.function);
            break;
          }
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    return evapor_model;
  }

} // namespace MeltPoolDG::Evaporation