#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

/**
 * This example is derived from
 *
 * Hardt, S., and F. Wondra. "Evaporation model for interfacial flows based on a continuum-field
 * representation of the source terms." Journal of Computational Physics 227.11 (2008): 5871-5895.
 *
 * and represents the film boiling example.
 */

namespace MeltPoolDG::Simulation::FilmBoiling
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  /**
   *  Initial level set field
   */
  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const double x_min,
                    const double x_max,
                    const double y_min,
                    const double y_interface,
                    const double lambda0)
      : Function<dim>()
      , x_min(x_min)
      , x_max(x_max)
      , y_min(y_min)
      , y_interface(y_interface)
      , lambda0(lambda0)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      double y_interface_disturbed =
        y_interface +
        lambda0 / 160. * std::cos(2 * numbers::PI * p[0] / lambda0); // according to Hardt

      Point<dim> lower_left  = dim == 1 ? Point<dim>(y_min) :
                               dim == 2 ? Point<dim>(x_min, y_min) :
                                          Point<dim>(x_min, x_min, y_min);
      Point<dim> upper_right = dim == 1 ? Point<dim>(y_interface_disturbed) :
                               dim == 2 ? Point<dim>(x_max, y_interface_disturbed) :
                                          Point<dim>(x_max, x_max, y_interface_disturbed);

      return -UtilityFunctions::CharacteristicFunctions::sgn(
        UtilityFunctions::DistanceFunctions::rectangular_manifold<dim>(p, lower_left, upper_right));
    }
    double x_min, x_max, y_min, y_interface, lambda0;
  };
  /**
   *  Initial temperature field
   */
  template <int dim>
  class InitialValuesTemperature : public Function<dim>
  {
  public:
    InitialValuesTemperature(const double T_max,
                             const double T_min,
                             const double y_interface,
                             const double lambda0)
      : Function<dim>()
      , T_max(T_max)
      , T_min(T_min)
      , y_interface(y_interface)
      , lambda0(lambda0)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      double y_interface_disturbed =
        y_interface + lambda0 / 160. * std::cos(2 * numbers::PI * p[0] / lambda0);

      if (p[dim - 1] > y_interface_disturbed)
        return T_min;
      else
        return T_max - (T_max - T_min) / y_interface_disturbed * p[dim - 1];
    }
    double T_max, T_min, y_interface, lambda0;
  };

  /**
   *      This class collects all relevant input data for the simulation.
   */
  template <int dim>
  class SimulationFilmBoiling : public SimulationBase<dim>
  {
  public:
    SimulationFilmBoiling(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
      , lambda0(
          2. * numbers::PI *
          std::sqrt(3. * this->parameters.flow.surface_tension_coefficient /
                    (this->parameters.base.gravity * (this->parameters.material.second.density -
                                                      this->parameters.material.first.density))))
      , x_max(lambda0 / 2.)
      , y_max(lambda0)
      , x_min(-x_max)
    {
      this->set_parameters();
    }

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

          GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                   subdivisions,
                                                                   bottom_left,
                                                                   top_right);
        }
      else
        {
          std::vector<unsigned int> subdivisions(dim, 1);
          subdivisions[dim - 1] *= 3;
          GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                    subdivisions,
                                                    bottom_left,
                                                    top_right);
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const types::boundary_id lower_bc = 1;
      const types::boundary_id upper_bc = 2;
      const types::boundary_id left_bc  = 3;
      const types::boundary_id right_bc = 4;


      if constexpr (dim == 1)
        {
          for (auto &cell : this->triangulation->cell_iterators())
            for (auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[0] == y_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[0] == y_max)
                    face->set_boundary_id(upper_bc);
                }
        }
      else if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[1] == y_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[1] == y_max)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == x_min)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
                }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");

      this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");

      this->attach_periodic_boundary_condition(left_bc, right_bc, 0);

      this->attach_dirichlet_boundary_condition(lower_bc,
                                                std::make_shared<Functions::ConstantFunction<dim>>(
                                                  this->parameters.evapor.boiling_temperature + 5.),
                                                "heat_transfer");

      // @todo: had to comment out the following line to not get a partitioner error --> maybe its
      //        the corner nodes have both -- dirichlet and PBC?
      this->attach_dirichlet_boundary_condition(
        lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1), "level_set");

      if (!this->parameters.base.do_simplex)
        this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(
        std::make_shared<InitialValuesLS<dim>>(x_min, x_max, y_min, 9. * lambda0 / 128., lambda0),
        "level_set");

      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");

      // boiling temperature at the interface
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(
                                       this->parameters.evapor.boiling_temperature + 5,
                                       this->parameters.evapor.boiling_temperature,
                                       9. * lambda0 / 128.,
                                       lambda0),
                                     "heat_transfer");
    }

  private:
    double       lambda0 = 0.0;
    double       x_max;
    double       y_max;
    const double x_min;
    const double y_min = 0.0;
  };
} // namespace MeltPoolDG::Simulation::FilmBoiling
