#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>

/**
 * This simulation is a reconstruction of the "moving droplet" example, as analyzed in
 *
 * Bussmann, M., Kothe, D. B., & Sicilian, J. M. (2002, January).
 * "Modeling high density ratio incompressible interfacial flows".
 * Fluids Engineering Division Summer Meeting (Vol. 36150, pp. 707-713).
 *
 * A droplet is subjected to a constant initial velocity, embedded in an ambient fluid
 * at rest.
 *
 * It serves as a benchmark to test whether the solver is capable of dealing
 * with high density ratios.
 *
 * @note: In this simulation, we considered an aspect ratio of an 1
 *        compared to Bussmann (2002) who considered 2.
 */

namespace MeltPoolDG::Simulation::MovingDroplet
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static double velocity = 1.0;

  template <int dim>
  class InitialLevelSet : public Function<dim>
  {
  public:
    InitialLevelSet(const double radius, const double eps)
      : Function<dim>()
      , distance(Point<dim>(), radius)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /* component */) const override
    {
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        -distance.value(p), eps);
    }

  private:
    const Functions::SignedDistance::Sphere<dim> distance;
    const double                                 eps;
  };

  template <int dim>
  class InitialVelocityField : public Function<dim>
  {
  public:
    InitialVelocityField(const Function<dim> &level_set)
      : Function<dim>(dim)
      , level_set(level_set)
    {}

    void
    vector_value(const Point<dim> &p, [[maybe_unused]] Vector<double> &values) const override
    {
      const auto hs = (level_set.value(p) + 1.) * 0.5;

      if (dim == 2)
        {
          values[0] = hs * velocity;
          values[1] = hs * velocity;
        }
      else
        AssertThrow(false, ExcMessage("Advection field for dim!=2 not implemented."));
    }

    const Function<dim> &level_set;
  };

  template <int dim>
  class SimulationMovingDroplet : public SimulationBase<dim>
  {
  public:
    SimulationMovingDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      AssertDimension(dim, 2);
    }

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("side length", side_length, "Side length of the quadratic domain.");
        prm.add_parameter("radius", radius, "Droplet radius.");
        prm.add_parameter("velocity", velocity, "Initial droplet velocity.");
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      GridGenerator::hyper_cube(*this->triangulation,
                                -side_length / 2,
                                side_length / 2,
                                true /*colorize*/);
    }

    void
    set_boundary_conditions() final
    {
      this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");

      this->attach_periodic_boundary_condition(0, 1, 0);
      if (dim > 1)
        this->attach_periodic_boundary_condition(2, 3, 1);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_field_conditions() final
    {
      // Here, the initial velocity field depends on the level set function. Thus, the diffuse
      // level set is given in the initial state on purpose.
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      AssertThrow(eps > 0, ExcNotImplemented());

      const auto ls = std::make_shared<InitialLevelSet<dim>>(radius, eps);
      this->attach_initial_condition(ls, "level_set");

      this->attach_initial_condition(std::make_shared<InitialVelocityField<dim>>(*ls),
                                     "navier_stokes_u");
    }

  private:
    double side_length = 400e-6;
    double radius      = 100e-6;
  };
} // namespace MeltPoolDG::Simulation::MovingDroplet
