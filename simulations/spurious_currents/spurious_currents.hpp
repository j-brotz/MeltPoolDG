#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/functions.hpp>

#include <iostream>

namespace MeltPoolDG::Simulation::SpuriousCurrents
{
  using namespace dealii;

  static double side_length = 5.0;
  static double radius      = 0.5;
  static double radius_2    = 0.75;

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
        prm.add_parameter("side length", side_length, "Side length of the quadratic domain.");
        prm.add_parameter("radius",
                          radius,
                          "Radius of (i) circle or (ii) of the first semi-axis of an ellipse.");
        prm.add_parameter("radius 2", radius_2, "Radius of second semi-axis of an ellipse.");
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
          GridGenerator::hyper_cube(*this->triangulation, -side_length / 2., side_length / 2);
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
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      AssertThrow(eps > 0, ExcNotImplemented());

      // introduce slightly non-symmetric geometry
      Point<dim> center;
      for (unsigned int d = 0; d < dim; ++d)
        center[d] = 0.02 + 0.01 * d;

      if (droplet_shape == "circle")
        {
          this->attach_initial_condition(
            std::make_shared<Functions::ChangedSignFunction<dim>>(
              std::make_shared<Functions::SignedDistance::Sphere<dim>>(center, radius)),
            "signed_distance");
        }
      else if (droplet_shape == "ellipse")
        {
          std::array<double, dim> radii;
          if constexpr (dim == 2)
            radii = {{radius_2, radius}};
          else
            AssertThrow(false, ExcNotImplemented());

          this->attach_initial_condition(
            std::make_shared<Functions::ChangedSignFunction<dim>>(
              std::make_shared<Functions::SignedDistance::Ellipsoid<dim>>(center, radii)),
            "signed_distance");
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
