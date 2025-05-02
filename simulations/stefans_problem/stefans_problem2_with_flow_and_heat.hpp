#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/finite_element_data.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <memory>
#include <string>
#include <vector>

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
  using namespace MeltPoolDG::Simulation;


  template <int dim>
  class InitialValuesLS : public dealii::Function<dim>
  {
  public:
    InitialValuesLS(const dealii::Point<dim> &lower_left, const dealii::Point<dim> &upper_right)
      : dealii::Function<dim>()
      , signed_distance(lower_left, upper_right)
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
    {
      return CharacteristicFunctions::sgn(-signed_distance.value(p));
    }

    dealii::Functions::SignedDistance::Rectangle<dim> signed_distance;
  };

  template <int dim>
  class InitialValuesTemperature : public dealii::Function<dim>
  {
  public:
    InitialValuesTemperature()
      : dealii::Function<dim>()
    {}

    double
    value(const dealii::Point<dim> &p, const unsigned int /*component*/) const override
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

  template <int dim, typename number>
  class SimulationStefansProblem2WithFlowAndHeat : public MeltPoolCase<dim, number>
  {
  public:
    SimulationStefansProblem2WithFlowAndHeat(std::string    parameter_file,
                                             const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP or dim == 1)
        {
#ifdef DEAL_II_WITH_METIS
          this->triangulation = std::make_shared<dealii::parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            dealii::Triangulation<dim>::none,
            false,
            dealii::parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
          AssertThrow(
            false,
            dealii::ExcMessage(
              "Missing Metis support of the deal.II installation. "
              "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
        }
      else
        {
          this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
            this->mpi_communicator);
        }

      // create mesh
      const dealii::Point<dim> bottom_left = dim == 1 ? dealii::Point<dim>(y_min) :
                                             dim == 2 ? dealii::Point<dim>(x_min, y_min) :
                                                        dealii::Point<dim>(x_min, x_min, y_min);
      const dealii::Point<dim> top_right   = dim == 1 ? dealii::Point<dim>(y_max) :
                                             dim == 2 ? dealii::Point<dim>(x_max, y_max) :
                                                        dealii::Point<dim>(x_max, x_max, y_max);

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          // create mesh
          std::vector<unsigned int> subdivisions(
            dim, 5 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;

          dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                           subdivisions,
                                                                           bottom_left,
                                                                           top_right);
        }
      else
        {
          dealii::GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const dealii::types::boundary_id lower_bc = 1;
      const dealii::types::boundary_id upper_bc = 2;
      const dealii::types::boundary_id left_bc  = 3;
      const dealii::types::boundary_id right_bc = 4;

      // lower part = liquid; upper part = gas
      this->attach_boundary_condition(lower_bc, "no_slip", "navier_stokes_u");
      this->attach_boundary_condition(upper_bc, "open", "navier_stokes_u");

      if (dim >= 2)
        {
          this->attach_boundary_condition(left_bc, "symmetry", "navier_stokes_u");
          this->attach_boundary_condition(right_bc, "symmetry", "navier_stokes_u");
        }

      this->attach_boundary_condition({lower_bc, std::make_shared<InitialValuesTemperature<dim>>()},
                                      "dirichlet",
                                      "heat_transfer");
      this->attach_boundary_condition(
        {lower_bc, std::make_shared<dealii::Functions::ConstantFunction<dim>>(1.0)},
        "dirichlet",
        "level_set");

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
              if (face->at_boundary())
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
              if (face->at_boundary())
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
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

    void
    set_field_conditions() final
    {
      const dealii::Point<dim> lower_left(dim == 1 ? dealii::Point<dim>(y_min) :
                                          dim == 2 ? dealii::Point<dim>(x_min, y_min) :
                                                     dealii::Point<dim>(x_min, x_min, y_min));
      const dealii::Point<dim> upper_right(
        dim == 1 ? dealii::Point<dim>(y_interface) :
        dim == 2 ? dealii::Point<dim>(x_max, y_interface) :
                   dealii::Point<dim>(x_max, x_max, y_interface));
      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(lower_left,
                                                                            upper_right),
                                     "level_set");
      this->attach_initial_condition(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
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
