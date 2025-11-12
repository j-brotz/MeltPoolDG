#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>

#include "../heat_transfer_case.hpp"

/**
 * This case implements the benchmark example: laser-induced heating of a 2D fixed melt pool surface
 * from
 *
 * [1] Much, N., Schreter-Fleischhacker, M., Munch, P., Kronbichler, M., Wall, W. A., & Meier, C.
 * "Improved accuracy of continuum surface flux models for metal additive manufacturing melt pool
 * simulations" Advanced Modeling and Simulation in Engineering Sciences, 11(1), 16. (2024)
 */
namespace MeltPoolDG::Simulation::FixedMeltPool
{
  BETTER_ENUM(LevelSetType, char, level_set, heaviside, signed_distance)

  /**
   * This dealii::Frunction implements the fixed melt pool geometry as any of the LevelSetType
   * representations. LevelSetType::signed_distance corresponds to eq. (37) in [1].
   */
  template <int dim, typename number>
  class FixedMeltPoolGeometry : public dealii::Function<dim, number>
  {
  public:
    FixedMeltPoolGeometry(const LevelSetType level_set_type, const number eps = 0.0);

    number
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override;

  private:
    number
    sd_level(const dealii::Point<dim> &p) const;

    number
    sd_pool(const dealii::Point<dim> &p) const;

    number
    sd_beads(const dealii::Point<dim> &p) const;

    inline bool
    is_inside_melt_pool(const dealii::Point<dim> &p) const;

    const number             y_level = 10e-6;
    const dealii::Point<dim> center{0.0, 0.0};
    const number             center_radius = 50e-6;
    const number             bead_radius   = 10e-6;

    const number melt_pool_radius = center_radius + bead_radius;
    const number bead_center_x    = center[0] + melt_pool_radius;

    const LevelSetType level_set_type;
    const number       eps;
  };
  ;

  template <int dim, typename number>
  class SimulationFixedMeltPool : public Heat::HeatTransferCase<dim, number>
  {
  private:
    const number a = 100e-6;
    std::string  grid_file;

    const number T_hat = 500.0;

    number interface_thickness = numbers::invalid_double;

  public:
    SimulationFixedMeltPool(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() override;

    void
    set_field_conditions() override;
  };

} // namespace MeltPoolDG::Simulation::FixedMeltPool
