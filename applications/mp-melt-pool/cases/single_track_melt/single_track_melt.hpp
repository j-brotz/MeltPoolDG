#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_in.h>

#include <memory>
#include <string>

#include "../../../mp-heat-transfer/heat_transfer_case.hpp"
#include "../../melt_pool_case.hpp"

namespace MeltPoolDG::Simulation::SingleTrackMelt
{
  using namespace dealii;

  /**
   * @brief Prescribed gas inlet velocity field for the single-track melt case.
   *
   * The velocity is defined only for three-dimensional simulations and returns
   * a constant velocity in the z-direction (=scan direction). All other velocity
   * components are zero.
   *
   * @tparam dim Spatial dimension.
   */
  template <int dim>
  class GasInletVelocity : public Function<dim>
  {
  public:
    /**
     * @brief Construct the gas inlet velocity function.
     */
    GasInletVelocity(const double inlet_velocity);

    /**
     * @brief Return the velocity value at a point for a selected component.
     *
     * @param p Evaluation point.
     * @param component Vector component to evaluate.
     * @return Constant inlet velocity for the z-component, zero otherwise.
     */
    double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    /**
     * inlet velocity in z-direction
     */
    const double inlet_velocity;
  };

  /**
   * @brief Initial signed-distance function for a flat interface.
   *
   * The signed distance is initialized as
   * \f[
   * \phi(\mathbf{x}) = -y + y_\mathrm{interface}.
   * \f]
   *
   * @tparam dim Spatial dimension.
   */
  template <int dim>
  class InitialSignedDistance : public Function<dim>
  {
  public:
    /**
     * @brief Construct the initial signed-distance function.
     *
     * @param y_interface Initial y-position of the interface.
     */
    InitialSignedDistance(const double y_interface);

    /**
     * @brief Evaluate the initial signed-distance value.
     *
     * @param p Evaluation point.
     * @param component Function component. Ignored for this scalar function.
     * @return Signed distance to the initial horizontal interface.
     */
    double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

  private:
    /**
     * @brief Initial y-position of the interface.
     */
    const double y_interface;
  };

  /**
   * @brief Simulation case for a single-track melt problem.
   *
   * This class defines the case-specific parameters, mesh generation or import,
   * boundary conditions, initial field conditions, and optional local refinement
   * regions for the single-track melt setup.
   *
   *
   * The implementation considers the single-track melt configuration presented in:
   *
   * Constantin Zenz, Peter S. Cook, Laszlo Vörös, and Andreas Otto,
   * "A critical comparison of one- and two-fluid approaches for the simulation
   * of laser-induced melt pool formation and vaporisation",
   * Discover Materials, 2025.
   *
   * DOI: 10.1007/s43939-025-00434-0
   * https://link.springer.com/article/10.1007/s43939-025-00434-0
   *
   * @tparam dim Spatial dimension.
   * @tparam Number Scalar number type.
   * @tparam CaseClass Base case class, for example MeltPoolCase or HeatTransferCase.
   */
  template <int dim, typename Number, typename CaseClass>
  class SimulationSingleTrackMelt : public CaseClass
  {
  public:
    /**
     * @brief Construct the single-track melt simulation case.
     *
     * @param parameter_file Path to the parameter file.
     * @param mpi_communicator MPI communicator used by the simulation.
     */
    SimulationSingleTrackMelt(std::string parameter_file, const MPI_Comm mpi_communicator);

    /**
     * @brief Declare case-specific parameters.
     *
     * @param prm Parameter handler used to declare and parse parameters.
     * @return Whether the parsed parameters should be printed.
     */
    bool
    add_case_specific_parameters(ParameterHandler &prm) override;

    /**
     * @brief Create or import the spatial discretization.
     *
     * Imports a gmsh mesh if available. Otherwise, creates a fallback
     * hyper-rectangle geometry.
     */
    void
    create_spatial_discretization() override;

    /**
     * @brief Attach boundary conditions and apply global/local mesh refinement.
     */
    void
    set_boundary_conditions() override;

    /**
     * @brief Attach initial conditions for velocity, signed distance, and temperature.
     */
    void
    set_field_conditions() override;

  private:
    /// Name of the gmsh mesh file to import.
    std::string mesh_file_name = "single-track-melt.msh";

    /// Initial temperature.
    double T_initial = 298.15;

    /// Gas inlet temperature.
    double inlet_temperature = 298.15;

    /// Gas inlet velocity in z-direction.
    double inlet_velocity = 100.0;

    /// Outlet pressure.
    double outlet_pressure = 0.0;

    /// Initial y-position of the gas/material interface.
    double y_interface = 0.0;

    /// Mutable time-step counter for case-specific operations.
    mutable unsigned int n_time_step = 0;

    /// Bottom-left point of the first local refinement region.
    dealii::Point<dim> local_refinement_1_bottom_left;

    /// Top-right point of the first local refinement region.
    dealii::Point<dim> local_refinement_1_top_right;

    /// Bottom-left point of the second local refinement region.
    dealii::Point<dim> local_refinement_2_bottom_left;

    /// Top-right point of the second local refinement region.
    dealii::Point<dim> local_refinement_2_top_right;

    /// Number of additional refinement cycles in the first refinement region.
    unsigned int n_additional_refinement_1 = 0;

    /// Number of additional refinement cycles in the second refinement region.
    unsigned int n_additional_refinement_2 = 0;
  };
} // namespace MeltPoolDG::Simulation::SingleTrackMelt
