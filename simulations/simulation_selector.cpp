#ifndef MELT_POOL_DG_DIM
#  define MELT_POOL_DG_DIM 1
#endif

#include <deal.II/base/exceptions.h>

// simulations
#include "advection_diffusion/advection_diffusion.hpp"
#include "evaporating_droplet/evaporating_droplet.hpp"
#include "evaporating_droplet/evaporating_droplet_with_heat.hpp"
#include "evaporating_droplet/evaporating_shell.hpp"
#include "film_boiling/film_boiling.hpp"
#include "flow_past_cylinder/flow_past_cylinder.hpp"
#include "melt_front_propagation/melt_front_propagation.hpp"
#include "moving_droplet/moving_droplet.hpp"
#include "oscillating_droplet/oscillating_droplet.hpp"
#include "powder_bed/powder_bed.hpp"
#include "radiative_transport/radiative_transport.hpp"
#include "recoil_pressure/recoil_pressure.hpp"
#include "reinit_circle/reinit_circle.hpp"
#include "rising_bubble/rising_bubble.hpp"
#include "rotating_bubble/rotating_bubble.hpp"
#include "simulation_selector.hpp"
#include "solidification_slab/solidification_slab.hpp"
#include "spurious_currents/spurious_currents.hpp"
#include "stefans_problem/stefans_problem.hpp"
#include "stefans_problem/stefans_problem1_with_flow_and_heat.hpp"
#include "stefans_problem/stefans_problem2_with_flow_and_heat.hpp"
#include "stefans_problem/stefans_problem_with_flow.hpp"
#include "thermo_capillary_droplet/thermo_capillary_droplet.hpp"
#include "thermo_capillary_two_droplets/thermo_capillary_two_droplets.hpp"
#include "unidirectional_heat_transfer/unidirectional_heat_transfer.hpp"
#include "vortex_bubble/vortex_bubble.hpp"
#include "zalesak_disk/zalesak_disk.hpp"

namespace MeltPoolDG::Simulation
{
  template <int dim>
  std::shared_ptr<SimulationBase<dim>>
  SimulationSelector<dim>::get_simulation(const ApplicationName simulation_name,
                                          const std::string     parameter_file,
                                          const MPI_Comm        mpi_communicator)
  {
    switch (simulation_name)
      {
        case ApplicationName::reinit_circle:
          return std::make_shared<ReinitCircle::SimulationReinit<dim>>(parameter_file,
                                                                       mpi_communicator);
        case ApplicationName::advection_diffusion:
          return std::make_shared<AdvectionDiffusion::SimulationAdvec<dim>>(parameter_file,
                                                                            mpi_communicator);
        case ApplicationName::rotating_bubble:
          return std::make_shared<RotatingBubble::SimulationRotatingBubble<dim>>(parameter_file,
                                                                                 mpi_communicator);
        case ApplicationName::flow_past_cylinder:
          return std::make_shared<FlowPastCylinder::SimulationFlowPastCylinder<dim>>(
            parameter_file, mpi_communicator);

        case ApplicationName::spurious_currents:
          return std::make_shared<SpuriousCurrents::SimulationSpuriousCurrents<dim>>(
            parameter_file, mpi_communicator);

        case ApplicationName::rising_bubble:
          return std::make_shared<RisingBubble::SimulationRisingBubble<dim>>(parameter_file,
                                                                             mpi_communicator);
        case ApplicationName::zalesak_disk:
          return std::make_shared<ZalesakDisk::SimulationZalesakDisk<dim>>(parameter_file,
                                                                           mpi_communicator);
        case ApplicationName::recoil_pressure:
          return std::make_shared<RecoilPressure::SimulationRecoilPressure<dim>>(parameter_file,
                                                                                 mpi_communicator);
        case ApplicationName::vortex_bubble:
          return std::make_shared<VortexBubble::SimulationVortexBubble<dim>>(parameter_file,
                                                                             mpi_communicator);
        case ApplicationName::stefans_problem:
          return std::make_shared<StefansProblem::SimulationStefansProblem<dim>>(parameter_file,
                                                                                 mpi_communicator);
        case ApplicationName::stefans_problem_with_flow:
          return std::make_shared<StefansProblemWithFlow::SimulationStefansProblemWithFlow<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::stefans_problem1_with_flow_and_heat:
          return std::make_shared<
            StefansProblem1WithFlowAndHeat::SimulationStefansProblem1WithFlowAndHeat<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::stefans_problem2_with_flow_and_heat:
          return std::make_shared<
            StefansProblem2WithFlowAndHeat::SimulationStefansProblem2WithFlowAndHeat<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::evaporating_droplet:
          return std::make_shared<EvaporatingDroplet::SimulationEvaporatingDroplet<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::evaporating_shell:
          return std::make_shared<EvaporatingShell::SimulationEvaporatingShell<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::evaporating_droplet_with_heat:
          return std::make_shared<
            EvaporatingDropletWithHeat::SimulationEvaporatingDropletWithHeat<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::unidirectional_heat_transfer:
          return std::make_shared<
            UnidirectionalHeatTransfer::SimulationUnidirectionalHeatTransfer<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::thermo_capillary_droplet:
          return std::make_shared<ThermoCapillaryDroplet::SimulationThermoCapillaryDroplet<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::thermo_capillary_two_droplets:
          return std::make_shared<
            ThermoCapillaryTwoDroplets::SimulationThermoCapillaryTwoDroplets<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::solidification_slab:
          return std::make_shared<SolidificationSlab::SimulationSolidificationSlab<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::film_boiling:
          return std::make_shared<FilmBoiling::SimulationFilmBoiling<dim>>(parameter_file,
                                                                           mpi_communicator);
        case ApplicationName::melt_front_propagation:
          return std::make_shared<MeltFrontPropagation::SimulationMeltFrontPropagation<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::oscillating_droplet:
          return std::make_shared<OscillatingDroplet::SimulationOscillatingDroplet<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::moving_droplet:
          return std::make_shared<MovingDroplet::SimulationMovingDroplet<dim>>(parameter_file,
                                                                               mpi_communicator);
        case ApplicationName::radiative_transport:
          return std::make_shared<RadiativeTransport::RadiativeTransportSimulation<dim>>(
            parameter_file, mpi_communicator);
        case ApplicationName::powder_bed:
          return std::make_shared<PowderBed::SimulationPowderBed<dim>>(parameter_file,
                                                                       mpi_communicator);
        /* add your simulation here*/
        default:
          AssertThrow(false,
                      ExcMessage(
                        "The input-file for your requested application does not exist. Abort ..."));
      }
  }

  template class SimulationSelector<MELT_POOL_DG_DIM>;
} // namespace MeltPoolDG::Simulation
