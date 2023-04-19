#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>

#include <cmath>
#include <iostream>

/**
 * TODO
 */

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
    IntensityBoundary()
      : Function<dim>(1)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      (void)p;
      // TODO
      AssertThrow(false, ExcNotImplemented());
      return 0.0;
    }
  };

  template <int dim>
  class HorizontalLevelSetHeaviside : public Function<dim>
  {
  public:
    HorizontalLevelSetHeaviside()
      : Function<dim>(1)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      const auto y = p[1];
      return UtilityFunctions::CharacteristicFunctions::heaviside(level - y, eps);
    }

  private:
    const double eps   = 0.1;
    const double level = 0.0;
  };

  template <int dim>
  class RadiativeTransportSimulation : public SimulationBase<dim>
  {
  public:
    RadiativeTransportSimulation(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

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

      const types::boundary_id upper_bc = 1;

      for (const auto &cell : this->triangulation->cell_iterators())
        for (auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              if (face->center()[dim - 1] == x_max)
                face->set_boundary_id(upper_bc);
            }

      this->attach_dirichlet_boundary_condition(upper_bc,
                                                std::make_shared<IntensityBoundary<dim>>(),
                                                "intensity");
    }

    void
    set_field_conditions() final
    {
      this->attach_source_field(std::make_shared<HorizontalLevelSetHeaviside<dim>>(), "heaviside");

      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                     "intensity");
    }
  };
} // namespace MeltPoolDG::Simulation::RadiativeTransport
