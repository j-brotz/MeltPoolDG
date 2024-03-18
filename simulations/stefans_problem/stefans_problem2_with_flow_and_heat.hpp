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
#include <meltpooldg/utilities/utility_functions.hpp>

/**
 * This example is derived from
 *
 * Hardt, S., and F. Wondra. "Evaporation model for interfacial flows based on a continuum-field
 * representation of the source terms." Journal of Computational Physics 227.11 (2008): 5871-5895.
 *
 * and represents a simplified example of the denoted "Stefan's Problem 2".
 */

namespace MeltPoolDG::Simulation::StefansProblem2WithFlowAndHeat
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;


  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const Point<dim> &lower_left, const Point<dim> &upper_right)
      : Function<dim>()
      , signed_distance(lower_left, upper_right)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return UtilityFunctions::CharacteristicFunctions::sgn(-signed_distance.value(p));
    }

    dealii::Functions::SignedDistance::Rectangle<dim> signed_distance;
  };

  template <int dim>
  class InitialValuesTemperature : public Function<dim>
  {
  public:
    InitialValuesTemperature()
      : Function<dim>()
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      const double T_bottom = 573.15;
      const double T_sat    = 373.15;
      if (p[dim - 1] >= 0.0001)
        return T_sat;
      else
        return T_bottom - (T_bottom - T_sat) / 1e-4 * p[dim - 1];
    }
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationStefansProblem2WithFlowAndHeat : public SimulationBase<dim>
  {
  public:
    SimulationStefansProblem2WithFlowAndHeat(std::string    parameter_file,
                                             const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
        {
#ifdef DEAL_II_WITH_METIS
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
          AssertThrow(
            false,
            ExcMessage(
              "Missing Metis support of the deal.II installation. "
              "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
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

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
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
          GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
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

      // lower part = liquid; upper part = gas
      this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
      this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");

      if (dim >= 2)
        {
          this->attach_symmetry_boundary_condition(left_bc, "navier_stokes_u");
          this->attach_symmetry_boundary_condition(right_bc, "navier_stokes_u");
        }

      this->attach_dirichlet_boundary_condition(lower_bc,
                                                std::make_shared<InitialValuesTemperature<dim>>(),
                                                "heat_transfer");
      this->attach_dirichlet_boundary_condition(
        lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(1.0), "level_set");

      /*
       *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
       *  due to the hyperbolic nature of the analyzed problem)
       *
                    open
              +---------------+
              |    ls=-1      |
              |               |
       sym    |               |  sym
              |               |
              |               |
              |    ls=1       |
              +---------------+
       *           fix/open
       */
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
    }

    void
    set_field_conditions() final
    {
      const Point<dim> lower_left((dim == 1) ? Point<dim>(y_min) :
                                  (dim == 2) ? Point<dim>(x_min, y_min) :
                                               Point<dim>(x_min, x_min, y_min));
      const Point<dim> upper_right((dim == 1) ? Point<dim>(y_interface) :
                                   (dim == 2) ? Point<dim>(x_max, y_interface) :
                                                Point<dim>(x_max, x_max, y_interface));
      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(lower_left,
                                                                            upper_right),
                                     "level_set");
      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(),
                                     "heat_transfer");
    }

  private:
    const double x_min       = 0.0;
    const double x_max       = 5.0e-4;
    const double y_min       = 0.0;
    const double y_max       = 0.5e-3;
    const double y_interface = 0.0001;
  };
} // namespace MeltPoolDG::Simulation::StefansProblem2WithFlowAndHeat
