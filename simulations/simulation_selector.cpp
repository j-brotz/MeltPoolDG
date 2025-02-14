#ifndef MELT_POOL_DG_DIM
#  define MELT_POOL_DG_DIM 1
#endif

#include <deal.II/base/exceptions.h>

// simulations
#include "evaporating_droplet/evaporating_droplet.hpp"
#include "evaporating_droplet/evaporating_droplet_with_heat.hpp"
#include "evaporating_droplet/evaporating_shell.hpp"
#include "film_boiling/film_boiling.hpp"
#include "flow_past_cylinder/flow_past_cylinder.hpp"
#include "melt_front_propagation/melt_front_propagation.hpp"
#include "moving_droplet/moving_droplet.hpp"
#include "oscillating_droplet/oscillating_droplet.hpp"
#include "recoil_pressure/recoil_pressure.hpp"
#include "rising_bubble/rising_bubble.hpp"
#include "simulation_selector.hpp"
#include "spurious_currents/spurious_currents.hpp"
#include "stefans_problem/stefans_problem1_with_flow_and_heat.hpp"
#include "stefans_problem/stefans_problem2_with_flow_and_heat.hpp"
#include "stefans_problem/stefans_problem_with_flow.hpp"
#include "thermo_capillary_droplet/thermo_capillary_droplet.hpp"
#include "thermo_capillary_two_droplets/thermo_capillary_two_droplets.hpp"

namespace MeltPoolDG::Simulation
{
  template <int dim>
  std::shared_ptr<MeltPoolCase<dim>>
  SimulationSelector<dim>::get_simulation(const std::string case_name,
                                          const std::string parameter_file,
                                          const MPI_Comm    mpi_communicator)
  {
    if (case_name == "flow_past_cylinder")
      return std::make_shared<FlowPastCylinder::SimulationFlowPastCylinder<dim>>(parameter_file,
                                                                                 mpi_communicator);
    else if (case_name == "spurious_currents")
      return std::make_shared<SpuriousCurrents::SimulationSpuriousCurrents<dim>>(parameter_file,
                                                                                 mpi_communicator);
    else if (case_name == "rising_bubble")
      return std::make_shared<RisingBubble::SimulationRisingBubble<dim>>(parameter_file,
                                                                         mpi_communicator);
    else if (case_name == "recoil_pressure")
      return std::make_shared<RecoilPressure::SimulationRecoilPressure<dim>>(parameter_file,
                                                                             mpi_communicator);
    else if (case_name == "stefans_problem_with_flow")
      return std::make_shared<StefansProblemWithFlow::SimulationStefansProblemWithFlow<dim>>(
        parameter_file, mpi_communicator);
    else if (case_name == "stefans_problem1_with_flow_and_heat")
      return std::make_shared<
        StefansProblem1WithFlowAndHeat::SimulationStefansProblem1WithFlowAndHeat<dim>>(
        parameter_file, mpi_communicator);
    else if (case_name == "stefans_problem2_with_flow_and_heat")
      return std::make_shared<
        StefansProblem2WithFlowAndHeat::SimulationStefansProblem2WithFlowAndHeat<dim>>(
        parameter_file, mpi_communicator);
    else if (case_name == "evaporating_droplet")
      return std::make_shared<EvaporatingDroplet::SimulationEvaporatingDroplet<dim>>(
        parameter_file, mpi_communicator);
    else if (case_name == "evaporating_shell")
      return std::make_shared<EvaporatingShell::SimulationEvaporatingShell<dim>>(parameter_file,
                                                                                 mpi_communicator);
    else if (case_name == "evaporating_droplet_with_heat")
      return std::make_shared<
        EvaporatingDropletWithHeat::SimulationEvaporatingDropletWithHeat<dim>>(parameter_file,
                                                                               mpi_communicator);
    else if (case_name == "thermo_capillary_droplet")
      return std::make_shared<ThermoCapillaryDroplet::SimulationThermoCapillaryDroplet<dim>>(
        parameter_file, mpi_communicator);
    else if (case_name == "thermo_capillary_two_droplets")
      return std::make_shared<
        ThermoCapillaryTwoDroplets::SimulationThermoCapillaryTwoDroplets<dim>>(parameter_file,
                                                                               mpi_communicator);
    else if (case_name == "film_boiling")
      return std::make_shared<FilmBoiling::SimulationFilmBoiling<dim>>(parameter_file,
                                                                       mpi_communicator);
    else if (case_name == "melt_front_propagation")
      return std::make_shared<MeltFrontPropagation::SimulationMeltFrontPropagation<dim>>(
        parameter_file, mpi_communicator);
    else if (case_name == "oscillating_droplet")
      return std::make_shared<OscillatingDroplet::SimulationOscillatingDroplet<dim>>(
        parameter_file, mpi_communicator);
    else if (case_name == "moving_droplet")
      return std::make_shared<MovingDroplet::SimulationMovingDroplet<dim>>(parameter_file,
                                                                           mpi_communicator);
    /* add your simulation here*/
    else
      {
        AssertThrow(false,
                    ExcMessage("The input-file for your requested case does not exist. Abort ..."));
        return nullptr;
      }
  }

  template class SimulationSelector<MELT_POOL_DG_DIM>;
} // namespace MeltPoolDG::Simulation
