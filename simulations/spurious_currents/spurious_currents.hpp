#pragma once

// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>

namespace MeltPoolDG::Simulation::SpuriousCurrents
{
  using namespace dealii;

  template <int dim>
  class InitialLevelSetCircle : public Function<dim>
  {
  public:
    InitialLevelSetCircle(const Point<dim> &center, const double radius, const double eps)
      : Function<dim>()
      , distance_sphere(center, radius)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        -distance_sphere.value(p), eps);
    }

  private:
    const Functions::SignedDistance::Sphere<dim> distance_sphere;
    double                                       eps;
  };

  template <int dim>
  class InitialLevelSetEllipse : public Function<dim>
  {
  public:
    InitialLevelSetEllipse(const Point<dim> &             center,
                           const std::array<double, dim> &radii,
                           const double                   eps)
      : Function<dim>()
      , distance_ellipse(center, radii)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        -distance_ellipse.value(p), eps);
    }

  private:
    const Functions::SignedDistance::Ellipsoid<dim> distance_ellipse;
    double                                          eps;
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationSpuriousCurrents : public SimulationBase<dim>
  {
  public:
    SimulationSpuriousCurrents(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("droplet shape",
                          droplet_shape,
                          "Shape of the droplet: circle or ellipse",
                          Patterns::Selection("circle|ellipse"));
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      if constexpr (dim == 2)
        {
          GridGenerator::hyper_cube(*this->triangulation, -2.5, 2.5);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    void
    set_boundary_conditions() override
    {
      this->attach_dirichlet_boundary_condition(
        0, std::make_shared<Functions::ConstantFunction<dim>>(-1), "level_set");
      this->attach_no_slip_boundary_condition(0, "navier_stokes_u");
      this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");
    }

    void
    set_field_conditions() override
    {
      double eps =
        UtilityFunctions::compute_initial_epsilon<dim>(this->parameters, *this->triangulation);

      AssertThrow(eps > 0, ExcNotImplemented());

      Point<dim> center;
      for (unsigned int d = 0; d < dim; ++d)
        center[d] = 0.02 + 0.01 * d;

      if (droplet_shape == "circle")
        {
          const double radius = 0.5;
          this->attach_initial_condition(
            std::make_shared<InitialLevelSetCircle<dim>>(center, radius, eps), "level_set");
        }
      else if (droplet_shape == "ellipse")
        {
          std::array<double, dim> radii;
          if constexpr (dim == 2)
            radii = {{0.75, 0.5}};
          else
            AssertThrow(false, ExcNotImplemented());

          this->attach_initial_condition(
            std::make_shared<InitialLevelSetEllipse<dim>>(center, radii, eps), "level_set");
        }
      else
        AssertThrow(false,
                    ExcMessage("Unknown droptlet shape: \"" + droplet_shape + "\"! Abort..."));
      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
    }

  private:
    std::string droplet_shape = "circle";
  };
} // namespace MeltPoolDG::Simulation::SpuriousCurrents
