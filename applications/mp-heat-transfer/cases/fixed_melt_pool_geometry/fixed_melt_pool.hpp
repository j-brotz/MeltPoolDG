#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <meltpooldg/level_set/level_set_type.hpp>
#include <meltpooldg/utilities/numbers.hpp>

#include <string>

#include "../../heat_transfer_case.hpp"

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
  /**
   * This dealii::Frunction implements the fixed melt pool geometry as any of the LevelSetType
   * representations. LevelSetType::signed_distance corresponds to eq. (37) in [1].
   */
  template <int dim, typename number>
  class FixedMeltPoolGeometry : public dealii::Function<dim, number>
  {
  public:
    /**
     * @param level_set_type level set type, options: level_set, heaviside, signed_distance
     * @param eps interface thickness parameter epsilon: interface_thickness = 6*eps
     *            only required for level set type level_set or heaviside
     */
    FixedMeltPoolGeometry(const LevelSet::LevelSetType level_set_type, const number eps = 0.0);

    /**
     * returns value corresponding to Point @param p
     */
    number
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override;

  private:
    /**
     * compute signed distance to top level of interface, i.e. @param y_level
     */
    number
    signed_distance_level(const dealii::Point<dim> &p) const;

    /**
     * compute signed distance to centre semicircle
     */
    number
    signed_distance_pool(const dealii::Point<dim> &p) const;

    /**
     * compute signed distance to beads
     */
    number
    signed_distance_beads(const dealii::Point<dim> &p) const;

    bool
    is_inside_melt_pool(const dealii::Point<dim> &p) const;

    /// top level of interface
    const number y_level = 10e-6;

    const dealii::Point<dim> centre{0.0, 0.0};
    const number             centre_radius = 50e-6;
    const number             bead_radius   = 10e-6;

    const number melt_pool_radius = centre_radius + bead_radius;
    const number bead_center_x    = centre[0] + melt_pool_radius;

    const LevelSet::LevelSetType level_set_type;
    const number                 eps;
  };

  /**
   * Simulation case for  benchmark example: laser-induced heating of a 2D fixed melt pool surface
   * [1]
   */
  template <int dim, typename number>
  class SimulationFixedMeltPool : public Heat::HeatTransferCase<dim, number>
  {
  private:
    /// length paramter
    const number a = 100e-6;

    /// optional starting grid file
    std::string grid_file;

    /// initial and dirichlet boundary temperature
    const number T_hat = 500.0;

    number interface_thickness = numbers::invalid_double;

  public:
    SimulationFixedMeltPool(std::string parameter_file, const MPI_Comm mpi_communicator);

    bool
    add_case_specific_parameters(dealii::ParameterHandler &prm) override;

    void
    create_spatial_discretization() override;

    void
    set_boundary_conditions() override;

    void
    set_field_conditions() override;
  };

} // namespace MeltPoolDG::Simulation::FixedMeltPool
