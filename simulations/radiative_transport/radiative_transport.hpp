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
BETTER_ENUM(InterfaceCase, char, straight, curved, particles)
namespace MeltPoolDG::Simulation::RadiativeTransport
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static constexpr double x_min = -1;
  static constexpr double x_max = 1;

  template <int dim>
  class IntensityBoundary : public Function<dim>
  {
  public:
    IntensityBoundary(const double power_in,const double radius_in,const  Point<dim, double> center_in)
      : Function<dim>(1)
      , power(power_in)
      , radius(radius_in)
      , center(center_in)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      const double peak_factor = 1. / (radius * radius * numbers::PI / 2);

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
  class HorizontalLevelSetHeaviside : public Function<dim>
  {
  public:
    HorizontalLevelSetHeaviside(std::pair<int,Point<dim>> interface_case_info_in)
      : Function<dim>(1),
      interface_case_info (interface_case_info_in)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      if (interface_case_info.first == 0)
        {
          const auto y = p[dim - 1];
          return UtilityFunctions::CharacteristicFunctions::heaviside(
                     level +
                       0.5 * x_max * ((this->get_time() < 10.5) ? this->get_time() : 10) / 10 - y,
                     eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid
        }
      else if (interface_case_info.first == 1)
        { // a rough non-template ellipse for the interface, because unlikely to be reused
          if constexpr (dim ==2)
            {
              const auto x = p [0];
              const auto y = p [1];
              return (( std::pow(x/interface_case_info.second[0],2.) + std::pow((y-x_max/3*((this->get_time() < 10.5) ? this->get_time() : 10))/interface_case_info.second[1],2.) < 1) ? 0.0 : 1.0);
            }
          else if constexpr (dim ==3 )
            {
              const auto x = p [0];
              const auto y = p [1];
              const auto z = p [2];
              return ( std::pow(x/interface_case_info.second[0],2.) + std::pow(y/interface_case_info.second[1],2.) + std::pow(z/interface_case_info.second[2],2.) < 1) ? 1.0 : 0.0;
            }
          else {
              AssertThrow(false, ExcImpossibleInDim(dim));
            }


          AssertThrow(false, ExcNotImplemented());
        }
      else if (interface_case_info.first == 2)
        { // TBD: particles
          AssertThrow(false, ExcNotImplemented());
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

  private:
    const double      eps   = 0.1;
    const double      level = 0.0;
    std::pair<int, Point<dim>> interface_case_info;
  };



  template <int dim>
  class RadiativeTransportSimulation : public SimulationBase<dim>
  {
  public:
    RadiativeTransportSimulation(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific parameters");
      {
        //        prm.add_parameter("json field - TBD ", private member - TBD, "desc - TBD");
        prm.add_parameter("a", a, "1st semi-axis of the ellipsoid (curved) heaviside");
        prm.add_parameter("b", b, "2nd semi-axis of the ellipsoid (curved) heaviside");
        prm.add_parameter("c", c, "3rd semi-axis of the ellipsoid (curved) heaviside");
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      if constexpr (dim == 1)
        {
          AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<Triangulation<dim>>();
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

      Point<dim> center_in;
      double     radius_in= x_max / 5;
      double     power_in = this->parameters.rte.power;
      center_in[dim-1] = x_max;

      this->attach_dirichlet_boundary_condition(upper_bc,
                                                std::make_shared<IntensityBoundary<dim>>(power_in,
                                                                                         radius_in,
                                                                                         center_in),
                                                "intensity");
    }

    void
    set_field_conditions() final
    {
      std::pair<int,Point<dim>> interface_case_info_in;
      if (interface_case == InterfaceCase::straight)
        interface_case_info_in.first = 0;
      else if (interface_case ==  InterfaceCase::curved)
        {
          interface_case_info_in.first = 1;
          (dim == 2)?interface_case_info_in.second = Point<dim>(a,b) :interface_case_info_in.second = Point<dim>(a,b,c);
        }
      else if (interface_case ==  InterfaceCase::particles)
        interface_case_info_in.first = 2;


      this->attach_source_field(std::make_shared<HorizontalLevelSetHeaviside<dim>>(
                                  interface_case_info_in),
                                "heaviside");
      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(), "intensity");
    }
  private:
    InterfaceCase interface_case = InterfaceCase::straight;
    double a;
    double b;
    double c;
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
