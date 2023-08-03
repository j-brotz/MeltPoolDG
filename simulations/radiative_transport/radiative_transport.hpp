#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>

#include <cmath>
#include <iostream>

/**
 * TODO: documentation
 */

namespace MeltPoolDG::Simulation::RadiativeTransport
{
  BETTER_ENUM(InterfaceCase, char, straight, curved, particles)

  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static constexpr double x_min = -1;
  static constexpr double x_max = 1;

  template <int dim>
  class IntensityBoundary : public Function<dim>
  {
  public:
    IntensityBoundary(const double             power_in,
                      const double             radius_in,
                      const Point<dim, double> center_in)
      : Function<dim>(1)
      , power(power_in)
      , radius(radius_in)
      , center(center_in)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      const double peak_factor = 1. / (radius * radius * numbers::PI * 0.5);

      const double r = center.distance(p);

      const double s          = r / radius;
      const double peak_power = power * peak_factor;
      return peak_power * std::exp(-2. * s * s);
      // TODO: verify if this works for 2D, 3D
    }

  private:
    const double             power;
    const double             radius;
    const Point<dim, double> center;
  };

  template <int dim>
  class LevelSetHeaviside : public Function<dim>
  {
  public:
    LevelSetHeaviside(const InterfaceCase interface_case_in,
                      const Point<dim> &  interface_case_info_in)
      : Function<dim>(1)
      , interface_case(interface_case_in)
      , interface_case_info(interface_case_info_in)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      if (interface_case == InterfaceCase::straight)
        {
          const auto y = p[dim - 1];
          return UtilityFunctions::CharacteristicFunctions::heaviside(
            level + 0.5 * x_max * ((this->get_time() < 10.5) ? this->get_time() : 10) / 10 - y,
            eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid
        }
      else if (interface_case == InterfaceCase::curved)
        { // a rough non-template ellipse for the interface, because unlikely to be reused
          if constexpr (dim == 2)
            {
              const auto x = p[0];
              const auto y = p[1];
              return (
                (std::pow(x / interface_case_info[0], 2.) +
                   std::pow((y - x_max / 3 * ((this->get_time() < 10.5) ? this->get_time() : 10)) /
                              interface_case_info[1],
                            2.) <
                 1) ?
                  0.0 :
                  1.0);
            }
          else if constexpr (dim == 3)
            {
              const auto x = p[0];
              const auto y = p[1];
              const auto z = p[2];
              return (std::pow(x / interface_case_info[0], 2.) +
                        std::pow(y / interface_case_info[1], 2.) +
                        std::pow(z / interface_case_info[2], 2.) <
                      1) ?
                       1.0 :
                       0.0;
            }
          else
            {
              AssertThrow(false, ExcImpossibleInDim(dim));
            }


          AssertThrow(false, ExcNotImplemented());
        }
      else if (interface_case == InterfaceCase::particles)
        { // TBD: particles
          AssertThrow(false, ExcNotImplemented());
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

  private:
    const double  eps   = 0.1;
    const double  level = 0.0;
    InterfaceCase interface_case;
    Point<dim>    interface_case_info;
  };



  template <int dim>
  class RadiativeTransportSimulation : public SimulationBase<dim>
  {
  public:
    RadiativeTransportSimulation(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      center_in[dim - 1] = x_max;
    }

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("ellipse-a",
                          ellipse_a,
                          "1st semi-axis of the ellipsoid (curved) heaviside");
        prm.add_parameter("ellipse-b",
                          ellipse_b,
                          "2nd semi-axis of the ellipsoid (curved) heaviside");
        prm.add_parameter("ellipse-c",
                          ellipse_c,
                          "3rd semi-axis of the ellipsoid (curved) heaviside");
        prm.add_parameter("source center", center_in, "location of the heat source center");
        prm.add_parameter("source radius", radius_in, "heat source radius");
        prm.add_parameter("interface case",
                          interface_case,
                          "kind of interface for this simulation");

        prm.add_parameter("power",
                          power_in,
                          "Sets the intensity scale of the laser source. Is a scalar value");
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      if constexpr (dim == 1)
        {
          this->triangulation =
            std::make_shared<parallel::shared::Triangulation<dim>>(this->mpi_communicator);
          // create mesh
          const Point<1> left(x_min);
          const Point<1> right(x_max);
          GridGenerator::hyper_rectangle(*this->triangulation, left, right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else if constexpr (dim == 2)
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
          // create mesh
          const Point<2> left(x_min, x_min);
          const Point<2> right(x_max, x_max);
          GridGenerator::hyper_rectangle(*this->triangulation, left, right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else if constexpr (dim == 3)
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
          const Point<3> lower_left(x_min, x_min, x_min);
          const Point<3> upper_right(x_max, x_max, x_max);
          GridGenerator::hyper_rectangle(*this->triangulation, lower_left, upper_right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else
        {
          AssertThrow(false, ExcImpossibleInDim(dim));
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const types::boundary_id upper_bc = 1;

      for (const auto &cell : this->triangulation->cell_iterators())
        for (auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              if (face->center()[dim - 1] == x_max)
                face->set_boundary_id(upper_bc);
            }

      this->attach_dirichlet_boundary_condition(upper_bc,
                                                std::make_shared<IntensityBoundary<dim>>(power_in,
                                                                                         radius_in,
                                                                                         center_in),
                                                "intensity");
    }

    void
    set_field_conditions() final
    {
      // pass simulation-specific parameters to the simulation class.
      // Done after json parsing, is relevant for heaviside
      if (interface_case == InterfaceCase::curved)
        {
          (dim == 2) ? interface_case_info_in = Point<dim>(ellipse_a, ellipse_b) :
                       interface_case_info_in = Point<dim>(ellipse_a, ellipse_b, ellipse_c);
        }

      // attach the heaviside function field
      this->attach_source_field(std::make_shared<LevelSetHeaviside<dim>>(interface_case,
                                                                         interface_case_info_in),
                                "heaviside");
    }

  private:
    InterfaceCase interface_case = InterfaceCase::straight;
    Point<dim>    interface_case_info_in;
    double        ellipse_a;
    double        ellipse_b;
    double        ellipse_c;
    Point<dim>    center_in;
    double        radius_in = x_max / 5.;
    double        power_in  = 0.1;
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
