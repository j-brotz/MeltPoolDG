#ifndef MELT_POOL_DG_DIM
#  define MELT_POOL_DG_DIM 1
#endif

#include <meltpooldg/simulations/simulation_selector.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>
// simulations
#include <meltpooldg/simulations/advection_diffusion/advection_diffusion.hpp>
#include <meltpooldg/simulations/evaporating_droplet/evaporating_droplet.hpp>
#include <meltpooldg/simulations/evaporating_droplet/evaporating_droplet_with_heat.hpp>
#include <meltpooldg/simulations/film_boiling/film_boiling.hpp>
#include <meltpooldg/simulations/flow_past_cylinder/flow_past_cylinder.hpp>
#include <meltpooldg/simulations/recoil_pressure/recoil_pressure.hpp>
#include <meltpooldg/simulations/reinit_circle/reinit_circle.hpp>
#include <meltpooldg/simulations/rising_bubble/rising_bubble.hpp>
#include <meltpooldg/simulations/rotating_bubble/rotating_bubble.hpp>
#include <meltpooldg/simulations/slotted_disc/slotted_disc.hpp>
#include <meltpooldg/simulations/solidification_slab/solidification_slab.hpp>
#include <meltpooldg/simulations/spurious_currents/spurious_currents.hpp>
#include <meltpooldg/simulations/stefans_problem/stefans_problem.hpp>
#include <meltpooldg/simulations/stefans_problem/stefans_problem1_with_flow_and_heat.hpp>
#include <meltpooldg/simulations/stefans_problem/stefans_problem2_with_flow_and_heat.hpp>
#include <meltpooldg/simulations/stefans_problem/stefans_problem_with_flow.hpp>
#include <meltpooldg/simulations/thermo_capillary_droplet/thermo_capillary_droplet.hpp>
#include <meltpooldg/simulations/thermo_capillary_two_droplets/thermo_capillary_two_droplets.hpp>
#include <meltpooldg/simulations/unidirectional_heat_transfer/unidirectional_heat_transfer.hpp>
#include <meltpooldg/simulations/vortex_bubble/vortex_bubble.hpp>

namespace MeltPoolDG::Simulation
{
  template <int dim>
  std::shared_ptr<SimulationBase<dim>>
  SimulationSelector<dim>::get_simulation(const std::string simulation_name,
                                          const std::string parameter_file,
                                          const MPI_Comm    mpi_communicator)
  {
    if (simulation_name == "reinit_circle")
      return std::make_shared<ReinitCircle::SimulationReinit<dim>>(parameter_file,
                                                                   mpi_communicator);
    else if (simulation_name == "advection_diffusion")
      return std::make_shared<AdvectionDiffusion::SimulationAdvec<dim>>(parameter_file,
                                                                        mpi_communicator);
    else if (simulation_name == "rotating_bubble")
      return std::make_shared<RotatingBubble::SimulationRotatingBubble<dim>>(parameter_file,
                                                                             mpi_communicator);
    else if (simulation_name == "flow_past_cylinder")
      return std::make_shared<FlowPastCylinder::SimulationFlowPastCylinder<dim>>(parameter_file,
                                                                                 mpi_communicator);

    else if (simulation_name == "spurious_currents")
      return std::make_shared<SpuriousCurrents::SimulationSpuriousCurrents<dim>>(parameter_file,
                                                                                 mpi_communicator);

    else if (simulation_name == "rising_bubble")
      return std::make_shared<RisingBubble::SimulationRisingBubble<dim>>(parameter_file,
                                                                         mpi_communicator);
    else if (simulation_name == "slotted_disc")
      return std::make_shared<SlottedDisc::SimulationSlottedDisc<dim>>(parameter_file,
                                                                       mpi_communicator);
    else if (simulation_name == "recoil_pressure")
      return std::make_shared<RecoilPressure::SimulationRecoilPressure<dim>>(parameter_file,
                                                                             mpi_communicator);
    else if (simulation_name == "vortex_bubble")
      return std::make_shared<VortexBubble::SimulationVortexBubble<dim>>(parameter_file,
                                                                         mpi_communicator);
    else if (simulation_name == "stefans_problem")
      return std::make_shared<StefansProblem::SimulationStefansProblem<dim>>(parameter_file,
                                                                             mpi_communicator);
    else if (simulation_name == "stefans_problem_with_flow")
      return std::make_shared<StefansProblemWithFlow::SimulationStefansProblemWithFlow<dim>>(
        parameter_file, mpi_communicator);
    else if (simulation_name == "stefans_problem1_with_flow_and_heat")
      return std::make_shared<
        StefansProblem1WithFlowAndHeat::SimulationStefansProblem1WithFlowAndHeat<dim>>(
        parameter_file, mpi_communicator);
    else if (simulation_name == "stefans_problem2_with_flow_and_heat")
      return std::make_shared<
        StefansProblem2WithFlowAndHeat::SimulationStefansProblem2WithFlowAndHeat<dim>>(
        parameter_file, mpi_communicator);
    else if (simulation_name == "evaporating_droplet")
      return std::make_shared<EvaporatingDroplet::SimulationEvaporatingDroplet<dim>>(
        parameter_file, mpi_communicator);
    else if (simulation_name == "evaporating_droplet_with_heat")
      return std::make_shared<
        EvaporatingDropletWithHeat::SimulationEvaporatingDropletWithHeat<dim>>(parameter_file,
                                                                               mpi_communicator);
    else if (simulation_name == "unidirectional_heat_transfer")
      return std::make_shared<
        UnidirectionalHeatTransfer::SimulationUnidirectionalHeatTransfer<dim>>(parameter_file,
                                                                               mpi_communicator);
    else if (simulation_name == "thermo_capillary_droplet")
      return std::make_shared<ThermoCapillaryDroplet::SimulationThermoCapillaryDroplet<dim>>(
        parameter_file, mpi_communicator);
    else if (simulation_name == "thermo_capillary_two_droplets")
      return std::make_shared<
        ThermoCapillaryTwoDroplets::SimulationThermoCapillaryTwoDroplets<dim>>(parameter_file,
                                                                               mpi_communicator);
    else if (simulation_name == "solidification_slab")
      return std::make_shared<SolidificationSlab::SimulationSolidificationSlab<dim>>(
        parameter_file, mpi_communicator);
    else if (simulation_name == "film_boiling")
      return std::make_shared<FilmBoiling::SimulationFilmBoiling<dim>>(parameter_file,
                                                                       mpi_communicator);

    /* add your simulation here*/
    else
      AssertThrow(false,
                  ExcMessage("The input-file for your requested application does not exist"));
  }

  template class SimulationSelector<MELT_POOL_DG_DIM>;
} // namespace MeltPoolDG::Simulation
