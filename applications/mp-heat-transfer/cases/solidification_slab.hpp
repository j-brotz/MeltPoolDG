#pragma once

#include <deal.II/base/mpi.h>

#include <meltpooldg/core/simulation_base.hpp>

#include <memory>
#include <string>

#include "../heat_transfer_case.hpp"


/**
 * This simulation represents a simple test example for heat transfer with solidification.
 * "A pseudo one-dimensional slab (material properties of ice / water) with length L = 1m is subject
 * to a fixed temperature Tˆ = 253K on its left edge at x = 0. The initial temperature in the whole
 * slab is T0 = 283K." [1]
 *
 * [1] Proell, S. D., Wall, W. A., & Meier, C. (2019). On phase change and latent heat models in
 * metal additive manufacturing process simulation. Advanced Modeling and Simulation in Engineering
 * Sciences, 7(1), 1--32. http://arxiv.org/abs/1906.06238
 *
 * Comini, G., Del Guidice, S., Lewis, R. W., & Zienkiewicz, O. C. (1974). Finite element solution
 * of non-linear heat conduction problems with special reference to phase change. International
 * Journal for Numerical Methods in Engineering, 8(3), 613–624.
 * https://doi.org/10.1002/nme.1620080314
 */

namespace MeltPoolDG::Simulation::SolidificationSlab
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;


  template <int dim>
  class SimulationSolidificationSlab : public Heat::HeatTransferCase<dim>
  {
  public:
    SimulationSolidificationSlab(std::string parameter_file, const MPI_Comm mpi_communicator);

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() final;

    void
    set_field_conditions() final;

    // for self-registration
    static SimulationCaseRegistrar<Heat::HeatTransferCase<dim>> registrar;
  };

  // for self-registration
  template <int dim>
  SimulationCaseRegistrar<MeltPoolDG::Heat::HeatTransferCase<dim>>
    SimulationSolidificationSlab<dim>::registrar(
      "solidification_slab",
      [](const std::string parameter_file, const MPI_Comm mpi_communicator) {
        return std::make_unique<SimulationSolidificationSlab<dim>>(parameter_file,
                                                                   mpi_communicator);
      });
} // namespace MeltPoolDG::Simulation::SolidificationSlab
