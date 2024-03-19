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

namespace MeltPoolDG::Simulation::StefansProblem
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

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
      const types::boundary_id upper_bc = 2 * (dim - 1) + 1;

      this->attach_dirichlet_boundary_condition(
        upper_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1.0), "level_set");
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
