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
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

/**
 * This example represents a benchmark from the NAFEMS collection.
 * It offers the following configurations dependent on the input parameters:
 *
 *  case 1: emissivity > 0, convection_coefficient = 0
 *             adiabatic
 *            +--------+
 *            |        |
 * T=const    |        |   radiation
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 2: emissivity = 0, convection_coefficient > 0
 *             adiabatic
 *            +--------+
 *            |        |
 * T=const    |        |   convection
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 3: emissivity > 0, convection_coefficient > 0
 *             adiabatic
 *            +--------+
 *            |        |
 * T=const    |        |   radiation + convection
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 4: emissivity = 0, convection_coefficient = 0
 *
 *             adiabatic
 *            +--------+
 *            |        |
 *  adiabatic | T0=lin |   adiabatic
 *            |        |
 *            +--------+
 *            adiabatic
 *
 */

namespace MeltPoolDG::Simulation::HeatTransferWithRadiation
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  /*
   *      This class collects all relevant input data for the level set simulation
   */
  template <int dim>
  class LinearTemp : public Function<dim>
  {
  public:
    LinearTemp()
      : Function<dim>(1, 0)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      return p[0];
    }
  };

  template <int dim>
  class SimulationHeatTransferWithRadiation : public SimulationBase<dim>
  {
  public:
    SimulationHeatTransferWithRadiation(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      this->set_parameters();
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
          const Point<2> left(x_min, 0.0);
          const Point<2> right(x_max, 0.1);
          GridGenerator::hyper_rectangle(*this->triangulation, left, right);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    void
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const types::boundary_id left_bc  = 1;
      const types::boundary_id right_bc = 2;

      if (this->parameters.heat.emissivity > 0.0)
        this->attach_radiation_boundary_condition(right_bc, "heat_conduction");

      if (this->parameters.heat.convection_coefficient > 0.0)
        this->attach_convection_boundary_condition(right_bc, "heat_conduction");

      if constexpr ((dim == 1) || (dim == 2))
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (auto &face : cell->face_iterators())
              if (face->at_boundary())
                {
                  if (face->center()[0] == x_min)
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
      if (this->parameters.heat.emissivity > 0.0 ||
          this->parameters.heat.convection_coefficient > 0.0)
        this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(1000),
                                       "heat_conduction");
      else
        this->attach_initial_condition(std::make_shared<LinearTemp<dim>>(), "heat_conduction");
    }

  private:
    const double x_min = 0.0;
    const double x_max = 0.1;
  };
} // namespace MeltPoolDG::Simulation::HeatTransferWithRadiation
