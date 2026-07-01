#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <meltpooldg/core/simulation_case_base.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#define MELTPOOLDG_REGISTER_MULTI_APP_CASE(CaseClass, ConcreteCaseClass, case_name, dim, number) \
  static bool case_name_is_registered_##dim =                                                    \
    SimulationCaseFactory<CaseClass<dim, number>>::register_simulation(                          \
      case_name, [](const std::string parameter_file, const MPI_Comm mpi_communicator) {         \
        return std::make_unique<ConcreteCaseClass<dim, number, CaseClass<dim, number>>>(         \
          parameter_file, mpi_communicator);                                                     \
      });

/**
 * This simulation is mainly meant to test the functionality of RTE
 */
namespace MeltPoolDG::Simulation::RadiativeTransport
{
  BETTER_ENUM(InterfaceCase, char, straight, single_powder_particle)

  using namespace MeltPoolDG::Simulation;

  template <int dim, typename number, typename Problem>
  class SimulationRadTrans : public Problem
  {
  public:
    SimulationRadTrans(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() final;

    void
    set_field_conditions() final;

  private:
    number                     domain_x_min = -1.;
    number                     domain_x_max = 1.;
    number                     domain_y_min = -1.;
    number                     domain_y_max = 1.;
    std::vector<unsigned int>  cell_repetitions;
    std::pair<number, number>  interface_case_info;
    number                     power_in = 0.1;
    dealii::Point<dim, number> center_in;
    number                     radius_in = domain_x_max / 5.;

    InterfaceCase interface_case = InterfaceCase::straight;

    number speed    = domain_x_max / 2;
    number end_time = 10.0;

    number powder_particle_offset = domain_x_max / 4.;
    number powder_particle_radius = domain_x_max / 6.;
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
