#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/core/simulation_base.hpp>

#include <memory>
#include <string>

#include "../heat_transfer_case.hpp"


/**
 * This simulation represents simple test examples for heat transfer with melt front propagation.
 * The problem is inspired by the Proell et al. [1] single track scan example.
 *
 * The slab has properties of Ti-6Al-4V, is initially below the solidus temperature and is subjected
 * to a Gusarov laser heat source [2] at x = 0.
 *
 * [1] Proell, S. D., Wall, W. A., & Meier, C. (2019). On phase change and latent heat models in
 * metal additive manufacturing process simulation. Advanced Modeling and Simulation in Engineering
 * Sciences, 7(1), 1-32. http://arxiv.org/abs/1906.06238
 *
 * [2] Gusarov, A. V., Yadroitsev, I., Bertrand, P., & Smurov, I. (2009). Model of Radiation and
 * Heat Transfer in Laser-Powder Interaction Zone at Selective Laser Melting. Journal of Heat
 * Transfer, 131(7), 1-10. https://doi.org/10.1115/1.3109245
 */

namespace MeltPoolDG::Simulation::MeltFrontPropagation
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  template <int dim>
  class SimulationMeltFrontPropagation : public Heat::HeatTransferCase<dim>
  {
  private:
    double x_min        = 0.0;
    double x_max        = 0.0;
    double y_min        = 0.0;
    double y_max        = 0.0;
    double z_min        = 0.0;
    double z_max        = 0.0;
    double T_0          = 0.0;
    bool   do_two_phase = false;

  public:
    SimulationMeltFrontPropagation(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override;

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
    SimulationMeltFrontPropagation<dim>::registrar(
      "melt_front_propagation",
      [](const std::string parameter_file, const MPI_Comm mpi_communicator) {
        return std::make_unique<SimulationMeltFrontPropagation<dim>>(parameter_file,
                                                                     mpi_communicator);
      });
} // namespace MeltPoolDG::Simulation::MeltFrontPropagation
