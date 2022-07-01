#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG::Simulation::StefansProblem
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;


  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const double x_min,
                    const double x_max,
                    const double y_min,
                    const double y_interface)
      : Function<dim>()
      , x_min(x_min)
      , x_max(x_max)
      , y_min(y_min)
      , y_interface(y_interface)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      Point<dim> lower_left = dim == 2 ? Point<dim>(x_min, y_min) : Point<dim>(x_min, x_min, y_min);
      Point<dim> upper_right =
        dim == 2 ? Point<dim>(x_max, y_interface) : Point<dim>(x_max, x_max, y_interface);

      return UtilityFunctions::CharacteristicFunctions::sgn(
        DistanceFunctions::rectangular_manifold<dim>(p, lower_left, upper_right));
    }
    double x_min, x_max, y_min, y_interface;
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationStefansProblem : public SimulationBase<dim>
  {
  public:
    SimulationStefansProblem(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.do_simplex || dim == 1)
        {
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }
      // create mesh
      const Point<dim> bottom_left = dim == 1   ? Point<dim>(y_min) :
                                     (dim == 2) ? Point<dim>(x_min, y_min) :
                                                  Point<dim>(x_min, x_min, y_min);
      const Point<dim> top_right   = dim == 1   ? Point<dim>(y_max) :
                                     (dim == 2) ? Point<dim>(x_max, y_max) :
                                                  Point<dim>(x_max, x_max, y_max);


      if (this->parameters.base.do_simplex)
        {
          // create mesh
          std::vector<unsigned int> subdivisions(
            dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;

          GridGenerator::subdivided_hyper_rectangle_with_simplices(
            *this->triangulation, subdivisions, bottom_left, top_right, true);
        }
      else
        {
          GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right, true);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() final
    {
      // faces in dim-1 direction
      const types::boundary_id lower_bc = 2 * (dim - 1);
      const types::boundary_id upper_bc = 2 * (dim - 1) + 1;

      if (this->parameters.evapor.ls_value_liquid == -1)
        {
          this->attach_dirichlet_boundary_condition(
            lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(1.0), "level_set");
        }
      else if (this->parameters.evapor.ls_value_liquid == 1)
        {
          this->attach_dirichlet_boundary_condition(
            upper_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1.0), "level_set");
        }
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<Functions::SignedDistance::Plane<dim>>(
                                       Point<dim>::unit_vector(dim - 1) * y_interface,
                                       -Point<dim>::unit_vector(dim - 1)),
                                     "signed_distance");
      this->attach_advection_field(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                   "level_set");
    }

  private:
    const double x_min       = 0.0;
    const double x_max       = 1.0;
    const double y_min       = 0.0;
    const double y_max       = 1.0;
    const double y_interface = 0.5;
  };
} // namespace MeltPoolDG::Simulation::StefansProblem
