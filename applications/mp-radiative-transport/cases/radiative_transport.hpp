#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/enum.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#define MELTPOOLDG_REGISTER_MULTI_APP_CASE(CaseClass, ConcreteCaseClass, case_name, dim)   \
  static bool case_name_is_registered_##dim =                                              \
    SimulationCaseFactory<CaseClass<dim>>::register_simulation(                            \
      case_name, [](const std::string parameter_file, const MPI_Comm mpi_communicator) {   \
        return std::make_unique<ConcreteCaseClass<dim, CaseClass<dim>>>(parameter_file,    \
                                                                        mpi_communicator); \
      });
/**
 * This simulation is mainly meant to test the functionality of RTE
 */
namespace MeltPoolDG::Simulation::RadiativeTransport
{
  BETTER_ENUM(InterfaceCase, char, straight, single_powder_particle)

  using namespace MeltPoolDG::Simulation;

  template <int dim, typename Problem>
  class SimulationRadTrans : public Problem
  {
  public:
    SimulationRadTrans(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() final;

    void
    set_field_conditions() final;

  private:
    double                    domain_x_min = -1.;
    double                    domain_x_max = 1.;
    double                    domain_y_min = -1.;
    double                    domain_y_max = 1.;
    std::vector<unsigned int> cell_repetitions;
    std::pair<double, double> interface_case_info;
    double                    power_in = 0.1;
    dealii::Point<dim>        center_in;
    double                    radius_in = domain_x_max / 5.;

    InterfaceCase interface_case = InterfaceCase::straight;

    double speed    = domain_x_max / 2;
    double end_time = 10.0;

    double powder_particle_offset = domain_x_max / 4.;
    double powder_particle_radius = domain_x_max / 6.;
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
