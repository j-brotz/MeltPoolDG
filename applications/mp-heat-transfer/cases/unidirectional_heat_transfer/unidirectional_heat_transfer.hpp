#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/simulation_base.hpp>

#include <memory>
#include <string>

#include "../../heat_transfer_case.hpp"


/**
 * This example represents a simple test example for heat transfer.
 * It offers the following configurations dependent on the input parameters:
 *
 *  case 1: emissivity > 0, convection_coefficient = 0
 *             adiabatic
 *            +--------+
 *            |        |
 * adiabatic  |        |   radiation
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 2: emissivity = 0, convection_coefficient > 0
 *             adiabatic
 *            +--------+
 *            |        |
 * adiabatic  |        |   convection
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 3: emissivity > 0, convection_coefficient > 0
 *             adiabatic
 *            +--------+
 *            |        |
 * adiabatic  |        |   radiation + convection
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 4: emissivity = 0, convection_coefficient = 0
 *
 *             adiabatic
 *            +--------+
 *            |        |
 *  adiabatic | T0=lin |   adiabatic
 *            |        |
 *            +--------+
 *            adiabatic
 *
 */

namespace MeltPoolDG::Simulation::UnidirectionalHeatTransfer
{
  using namespace MeltPoolDG::Simulation;

  template <int dim, typename number>
  class SimulationUnidirectionalHeatTransfer : public Heat::HeatTransferCase<dim, number>
  {
  private:
    bool   do_solidification = false;
    bool   do_two_phase      = false;
    number velocity          = 0.0;

  public:
    SimulationUnidirectionalHeatTransfer(std::string    parameter_file,
                                         const MPI_Comm mpi_communicator);

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() final;

    void
    set_field_conditions() final;
  };
} // namespace MeltPoolDG::Simulation::UnidirectionalHeatTransfer
