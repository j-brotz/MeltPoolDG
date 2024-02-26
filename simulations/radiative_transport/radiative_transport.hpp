#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include <meltpooldg/heat/laser_utilities.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/enum.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

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

  // boundary condition for RTE
  template <int dim>
  class IntensityBoundary : public Function<dim>
  {
  public:
    IntensityBoundary(const double                  power_in,
                      const double                  radius_in,
                      const Point<dim, double>     &laser_position_in,
                      const Tensor<1, dim, double> &laser_direction_in)
      : Function<dim>(1)
      , direction(laser_direction_in)
      , gauss(power_in, radius_in, laser_position_in, direction)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      return gauss.compute_intensity(p);
    }

  private:
    const Tensor<1, dim, double>                             direction;
    const Heat::GaussProjectionIntensityProfile<dim, double> gauss;
  };

  template <int dim>
  class LevelSetHeaviside : public Function<dim>
  {
  public:
    LevelSetHeaviside(const InterfaceCase              interface_case_in,
                      const std::pair<double, double> &interface_case_info_in,
                      const double                     epsilon_cell_in)
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
          // say we inherit from a straight interface case
          const auto y = p[dim - 1];

          double straight_value = UtilityFunctions::CharacteristicFunctions::heaviside(
            level - y,
            eps); // 0 side of H() stands for gas, 1 side of H() stands for liquid

          Point<dim> sphere_center;
          sphere_center[dim - 1] = interface_case_info.first;

          // now add a power_particle with gradient:
          const Functions::SignedDistance::Sphere<dim> distance_sphere(sphere_center,
                                                                       interface_case_info.second);
          double                                       power_particle_value =
            UtilityFunctions::CharacteristicFunctions::heaviside(-distance_sphere.value(p), eps);
          return std::max(straight_value, power_particle_value);
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
    const double              eps                 = 0.1;
    const double              level               = 0.0;
    InterfaceCase             interface_case      = InterfaceCase::straight;
    std::pair<double, double> interface_case_info = std::pair<double, double>(0., 0.);
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
          const Point<3> lower_left(x_min, x_min, x_min);
          const Point<3> upper_right(x_max, x_max, x_max);
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
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

      this->attach_dirichlet_boundary_condition(
        upper_bc,
        std::make_shared<IntensityBoundary<dim>>(
          power_in, radius_in, center_in, -dealii::Point<dim>::unit_vector(dim - 1)),
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
          interface_case_info_in =
            std::pair<double, double>(powder_particle_offset, powder_particle_radius);
        }

      // determine the interface epsilon parameter from minimum mesh size
      double       thickness_scale_factor = 2.5;
      const double epsilon_cell           = GridTools::minimal_cell_diameter(*this->triangulation) /
                                  std::sqrt(dim) * thickness_scale_factor;

      // attach the prescribed heaviside function field
      this->attach_initial_condition(std::make_shared<LevelSetHeaviside<dim>>(
                                       interface_case, interface_case_info_in, epsilon_cell),
                                     "prescribed_heaviside");

      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(), "intensity");

      // this is only used to TODO generate the comparison solution with the gaussian laser
      if (this->parameters.base.problem_name == ProblemType::heat_transfer)
        {
          // attach dummy initial conditions for the heat transfer operation
          this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(),
                                         "heat_transfer");
        }
    }

  private:
    InterfaceCase             interface_case = InterfaceCase::straight;
    std::pair<double, double> interface_case_info_in;
    Point<dim>                center_in;
    double                    radius_in              = x_max / 5.;
    double                    powder_particle_offset = x_max / 4.;
    double                    powder_particle_radius = x_max / 6.;
    double                    power_in               = 0.1;
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
