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
 */

namespace MeltPoolDG::Simulation::HeatTransferWithRadiation
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  /*
   *      This class collects all relevant input data for the level set simulation
   */

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
      if (this->parameters.base.do_simplex)
        {
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (::Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      if constexpr (dim == 1)
        {
          // create mesh
          const Point<1> left(x_min);
          const Point<1> right(x_max);
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

      this->attach_dirichlet_boundary_condition(
        left_bc, std::make_shared<Functions::ConstantFunction<dim>>(1000.0), "heat_conduction_T");

      this->attach_radiation_boundary_condition(right_bc, "heat_conduction_T");

      if constexpr (dim == 1)
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
      this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(1000),
                                     "heat_conduction_T");
    }

  private:
    const double x_min = 0.0;
    const double x_max = 0.1;
  };
} // namespace MeltPoolDG::Simulation::HeatTransferWithRadiation
