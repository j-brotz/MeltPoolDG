#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/melt_pool/powder_bed.hpp>

#include <memory>
#include <string>
#include <vector>

#include "../heat_transfer_case.hpp"


namespace MeltPoolDG::Simulation::PowderBed
{
  template <int dim>
  class SimulationPowderBed : public Heat::HeatTransferCase<dim>
  {
  private:
    double                    domain_x_min = 0;
    double                    domain_x_max = 0;
    double                    domain_y_min = 0;
    double                    domain_y_max = 0;
    double                    domain_z_min = 0;
    double                    domain_z_max = 0;
    std::vector<unsigned int> cell_repetitions;
    double                    T_initial = 500;
    MeltPool::PowderBedData   powder_bed_data;

  public:
    SimulationPowderBed(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() override;

    void
    set_field_conditions() override;

    // for self-registration
    static SimulationCaseRegistrar<Heat::HeatTransferCase<dim>> registrar;
  };

  // for self-registration
  template <int dim>
  SimulationCaseRegistrar<MeltPoolDG::Heat::HeatTransferCase<dim>>
    SimulationPowderBed<dim>::registrar(
      "powder_bed",
      [](const std::string parameter_file, const MPI_Comm mpi_communicator) {
        return std::make_unique<SimulationPowderBed<dim>>(parameter_file, mpi_communicator);
      });
} // namespace MeltPoolDG::Simulation::PowderBed
