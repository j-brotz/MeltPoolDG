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
  BETTER_ENUM(InterfaceCase, char, straight, single_powder_particle, powderbed)

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
                      const Point<dim>   &interface_case_info_in,
                      const double        epsilon_cell_in)
      : Function<dim>(1)
      , eps(epsilon_cell_in)
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
      else if (interface_case == InterfaceCase::single_powder_particle)
        {
          if constexpr (dim == 2)
            {
              // say we inherit from a straight interface case
              const auto y = p[1];

              double straight_value = UtilityFunctions::CharacteristicFunctions::heaviside(
                level - y,
                eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid

              // now add a power_particle with gradient:
              const Functions::SignedDistance::Sphere<dim> distance_sphere(
                Point<dim>(0, interface_case_info[0]), interface_case_info[1]);
              double power_particle_value =
                UtilityFunctions::CharacteristicFunctions::heaviside(-distance_sphere.value(p),
                                                                     eps);
              return std::max(straight_value, power_particle_value);
            }
        }
      else if (interface_case == InterfaceCase::powderbed)
        {
          AssertThrow(false, ExcNotImplemented());
          return 0.0;
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
          return 0.0;
        }
      return 0.0;
    }

  private:
    const double  eps                 = 0.1;
    const double  level               = 0.0;
    InterfaceCase interface_case      = InterfaceCase::straight;
    Point<dim>    interface_case_info = Point<dim>();
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
        prm.add_parameter("source center", center_in, "location of the heat source center");
        prm.add_parameter("source radius", radius_in, "heat source radius");
        prm.add_parameter("interface case",
                          interface_case,
                          "kind of interface for this simulation");
        prm.add_parameter("power",
                          power_in,
                          "Sets the intensity scale of the laser source. Is a scalar value");
        prm.add_parameter("power_particle radius",
                          powder_particle_radius,
                          "hanging power_particle radius");
        prm.add_parameter("powder particle offset",
                          powder_particle_offset,
                          "hanging power_particle offset from [dim-1] = 0 plane");
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
      if (interface_case == InterfaceCase::single_powder_particle)
        {
          // TBD
          interface_case_info_in = Point<dim>(powder_particle_offset, powder_particle_radius);
        }

      // determine the interface epsilon parameter from minimum mesh size
      double       thickness_scale_factor = 2.5;
      const double epsilon_cell           = GridTools::minimal_cell_diameter(*this->triangulation) /
                                  std::sqrt(dim) * thickness_scale_factor;

      // attach the heaviside function field
      this->attach_source_field(std::make_shared<LevelSetHeaviside<dim>>(interface_case,
                                                                         interface_case_info_in,
                                                                         epsilon_cell),
                                "heaviside");
      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(), "intensity");
    }

  private:
    InterfaceCase interface_case = InterfaceCase::straight;
    Point<dim>    interface_case_info_in;
    Point<dim>    center_in;
    double        radius_in              = x_max / 5.;
    double        powder_particle_offset = x_max / 4.;
    double        powder_particle_radius = x_max / 20.;
    double        power_in               = 0.1;
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
