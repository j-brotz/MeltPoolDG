#include <meltpooldg/phase_change/evaporation_model_factory.hpp>
//
#include <deal.II/base/exceptions.h>

#include <meltpooldg/phase_change/evaporation_model_constant.hpp>
#include <meltpooldg/phase_change/evaporation_model_hardt_wondra.hpp>
#include <meltpooldg/phase_change/evaporation_model_recoil_pressure.hpp>
#include <meltpooldg/phase_change/evaporation_model_saturated_vapor_pressure.hpp>


namespace MeltPoolDG::Evaporation
{
  template <typename number>
  std::unique_ptr<EvaporationModelBase<number>>
  get_evaporation_model(const EvaporationData<number> &evapor_data,
                        const MaterialData<number>    &material_data)
  {
    std::unique_ptr<EvaporationModelBase<number>> evapor_model;

    switch (evapor_data.evaporative_mass_flux_model)
      {
          case EvaporationModelType::recoil_pressure: {
            evapor_model = std::make_unique<EvaporationModelRecoilPressure<number>>(
              evapor_data.recoil,
              material_data.boiling_temperature,
              material_data.molar_mass,
              material_data.latent_heat_of_evaporation);
            break;
          }
          case EvaporationModelType::saturated_vapor_pressure: {
            evapor_model = std::make_unique<EvaporationModelSaturatedVaporPressure<number>>(
              evapor_data.recoil,
              material_data.boiling_temperature,
              material_data.molar_mass,
              material_data.latent_heat_of_evaporation);
            break;
          }
          case EvaporationModelType::hardt_wondra: {
            evapor_model = std::make_unique<EvaporationModelHardtWondra<number>>(
              evapor_data.hardt_wondra.coefficient,
              material_data.latent_heat_of_evaporation,
              material_data.gas.density,
              material_data.molar_mass,
              material_data.boiling_temperature);
            break;
          }
          case EvaporationModelType::analytical: {
            evapor_model =
              std::make_unique<EvaporationModelConstant<number>>(evapor_data.analytical.function);
            break;
          }
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    return evapor_model;
  }



  template std::unique_ptr<EvaporationModelBase<double>>
  get_evaporation_model(const EvaporationData<double> &, const MaterialData<double> &);
} // namespace MeltPoolDG::Evaporation