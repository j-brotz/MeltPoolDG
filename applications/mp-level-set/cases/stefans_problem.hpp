#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/simulation_case_base.hpp>

#include <cmath>
#include <iostream>
#include <string>

#include "../level_set_case.hpp"

namespace MeltPoolDG::Simulation::StefansProblem
{
  using namespace MeltPoolDG::Simulation;

  template <int dim, typename number>
  class SimulationStefansProblem : public LevelSet::LevelSetCase<dim, number>
  {
  public:
    SimulationStefansProblem(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::LevelSetCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
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
      const dealii::Point<dim, number> bottom_left =
        dim == 1 ? dealii::Point<dim, number>(y_min) :
        dim == 2 ? dealii::Point<dim, number>(x_min, y_min) :
                   dealii::Point<dim, number>(x_min, x_min, y_min);
      const dealii::Point<dim, number> top_right =
        dim == 1 ? dealii::Point<dim, number>(y_max) :
        dim == 2 ? dealii::Point<dim, number>(x_max, y_max) :
                   dealii::Point<dim, number>(x_max, x_max, y_max);


      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          // create mesh
          std::vector<unsigned int> subdivisions(
            dim, 5 * dealii::Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;

          dealii::GridGenerator::subdivided_hyper_rectangle_with_simplices(
            *this->triangulation, subdivisions, bottom_left, top_right, true);
        }
      else
        {
          dealii::GridGenerator::hyper_rectangle(*this->triangulation,
                                                 bottom_left,
                                                 top_right,
                                                 true);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() final
    {
      // faces in dim-1 direction
      const dealii::types::boundary_id upper_bc = 2 * (dim - 1) + 1;

      this->attach_boundary_condition(
        {upper_bc, std::make_shared<dealii::Functions::ConstantFunction<dim>>(-1.0)},
        "dirichlet",
        "level_set");
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(
        std::make_shared<dealii::Functions::SignedDistance::Plane<dim>>(
          dealii::Point<dim, number>::unit_vector(dim - 1) * y_interface,
          -dealii::Point<dim, number>::unit_vector(dim - 1)),
        "signed_distance");
      this->attach_field_function(std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim),
                                  "prescribed_velocity",
                                  "level_set");
    }

  private:
    const number x_min       = 0.0;
    const number x_max       = 1.0;
    const number y_min       = 0.0;
    const number y_max       = 1.0;
    const number y_interface = 0.5;
  };
} // namespace MeltPoolDG::Simulation::StefansProblem
