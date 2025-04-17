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
  template <int dim, typename number>
  std::unique_ptr<MeltPoolCase<dim, number>>
  SimulationSelector<dim, number>::get_simulation(const std::string case_name,
                                                  const std::string parameter_file,
                                                  const MPI_Comm    mpi_communicator)
  {
    if (case_name == "flow_past_cylinder")
      return std::make_unique<FlowPastCylinder::SimulationFlowPastCylinder<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "spurious_currents")
      return std::make_unique<SpuriousCurrents::SimulationSpuriousCurrents<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "rising_bubble")
      return std::make_unique<RisingBubble::SimulationRisingBubble<dim, number>>(parameter_file,
                                                                                 mpi_communicator);
    else if (case_name == "recoil_pressure")
      return std::make_unique<RecoilPressure::SimulationRecoilPressure<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "stefans_problem_with_flow")
      return std::make_unique<
        StefansProblemWithFlow::SimulationStefansProblemWithFlow<dim, number>>(parameter_file,
                                                                               mpi_communicator);
    else if (case_name == "stefans_problem1_with_flow_and_heat")
      return std::make_unique<
        StefansProblem1WithFlowAndHeat::SimulationStefansProblem1WithFlowAndHeat<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "stefans_problem2_with_flow_and_heat")
      return std::make_unique<
        StefansProblem2WithFlowAndHeat::SimulationStefansProblem2WithFlowAndHeat<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "evaporating_droplet")
      return std::make_unique<EvaporatingDroplet::SimulationEvaporatingDroplet<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "evaporating_shell")
      return std::make_unique<EvaporatingShell::SimulationEvaporatingShell<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "evaporating_droplet_with_heat")
      return std::make_unique<
        EvaporatingDropletWithHeat::SimulationEvaporatingDropletWithHeat<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "thermo_capillary_droplet")
      return std::make_unique<
        ThermoCapillaryDroplet::SimulationThermoCapillaryDroplet<dim, number>>(parameter_file,
                                                                               mpi_communicator);
    else if (case_name == "thermo_capillary_two_droplets")
      return std::make_unique<
        ThermoCapillaryTwoDroplets::SimulationThermoCapillaryTwoDroplets<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "film_boiling")
      return std::make_unique<FilmBoiling::SimulationFilmBoiling<dim, number>>(parameter_file,
                                                                               mpi_communicator);
    else if (case_name == "melt_front_propagation")
      return std::make_unique<MeltFrontPropagation::SimulationMeltFrontPropagation<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "oscillating_droplet")
      return std::make_unique<OscillatingDroplet::SimulationOscillatingDroplet<dim, number>>(
        parameter_file, mpi_communicator);
    else if (case_name == "moving_droplet")
      return std::make_unique<MovingDroplet::SimulationMovingDroplet<dim, number>>(
        parameter_file, mpi_communicator);
    /* add your simulation here*/
    else
      {
        AssertThrow(false,
                    dealii::ExcMessage(
                      "The input-file for your requested case does not exist. Abort ..."));
        return nullptr;
      }
  }

  template class SimulationSelector<MELT_POOL_DG_DIM, double>;
} // namespace MeltPoolDG::Simulation
