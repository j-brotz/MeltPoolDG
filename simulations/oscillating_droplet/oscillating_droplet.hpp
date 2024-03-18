#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>

// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

// c++
#include <cmath>
#include <iostream>

/**
 * This simulation is a reconstruction of the "Oscillation of liquid droplet surrounded by gas
 * atmosphere" by Meier et al. [1]. In a quadratic gaseous domain of side length 400µm a droplet
 * starts of with an elliptical shape with a larger semiaxis of 3/2 * 100µm and a smaller semiaxis
 * of 2/3 * 100µm. Due to surface tension forces the droplet starts to oscillate.
 *
 * [1] Meier, C., Fuchs, S. L., Hart, A. J., & Wall, W. A. (2021). A novel smoothed particle
 * hydrodynamics formulation for thermo-capillary phase change problems with focus on metal additive
 * manufacturing melt pool modeling. Computer Methods in Applied Mechanics and Engineering, 381,
 * 113812. https://doi.org/10.1016/j.cma.2021.113812
 */

namespace MeltPoolDG::Simulation::OscillatingDroplet
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  template <int dim>
  class InitialLevelSet : public Function<dim>
  {
  public:
    InitialLevelSet(const std::array<double, dim> &radii, const double eps)
      : Function<dim>()
      , distance_ellipse(Point<dim>(), radii)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        -distance_ellipse.value(p), eps);
    }

  private:
    const Functions::SignedDistance::Ellipsoid<dim> distance_ellipse;
    const double                                    eps;
  };

  template <int dim>
  class SimulationOscillatingDroplet : public SimulationBase<dim>
  {
  public:
    SimulationOscillatingDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      AssertDimension(dim, 2);
    }

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("side length", side_length, "Side length of the quadratic domain.");
        prm.add_parameter("reference radius", reference_radius, "Reference radius.");
        prm.add_parameter("elliptical deviation",
                          elliptical_deviation,
                          "Deviation of the elliptical semiaxes from the reference radius.");
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      GridGenerator::hyper_cube(*this->triangulation, -side_length / 2, side_length / 2);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() final
    {
      this->attach_no_slip_boundary_condition(0, "navier_stokes_u");
      this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");
    }

    void
    set_field_conditions() final
    {
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      AssertThrow(eps > 0, ExcNotImplemented());

      std::array<double, dim> radii;
      if constexpr (dim == 2)
        radii = {
          {reference_radius * elliptical_deviation, reference_radius / elliptical_deviation}};
      else
        AssertThrow(false, ExcNotImplemented());

      this->attach_initial_condition(std::make_shared<InitialLevelSet<dim>>(radii, eps),
                                     "level_set");

      this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                       std::make_shared<Functions::ZeroFunction<dim>>(dim)),
                                     "navier_stokes_u");
    }

  private:
    double side_length          = 400e-6;
    double reference_radius     = 100e-6;
    double elliptical_deviation = 3. / 2.;
  };
} // namespace MeltPoolDG::Simulation::OscillatingDroplet
