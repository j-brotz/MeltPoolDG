#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/utility_functions.hpp>

#include "../compressible_flow_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  template <int dim>
  class InitialFlowField : public Function<dim>
  {
  public:
    explicit InitialFlowField()
      : Function<dim>(dim + 2)
    {
      Assert(dim == 2 or dim == 3, ExcNotImplemented());
    }

    double
    value(const Point<dim> &, const unsigned int component) const final
    {
      if (component == 0)
        return 1.;
      else if (component == dim + 1)
        return 3.017857142;
      else
        return 0.;
    }
  };

  template <int dim>
  class UnfittedObjectVelocity : public Function<dim>
  {
  public:
    explicit UnfittedObjectVelocity()
      : Function<dim>(dim)
    {}

    double
    value(const Point<dim> &, const unsigned int component) const final
    {
      if (component == 0)
        return 0.1;
      else
        return 0.;
    }
  };

  template <int dim>
  class MovingLevelSet : public Function<dim>
  {
  public:
    explicit MovingLevelSet(const double time)
      : Function<dim>(1, time)
    {}

    double
    value(const Point<dim> &p, const unsigned int /* component */) const override
    {
      const double t = this->get_time();
      Point<dim>   center;
      center[0] = 0.20048 + 0.1 * t;
      for (unsigned int d = 1; d < dim; ++d)
        center[d] = 0.24;

      const Functions::SignedDistance::Sphere<dim> distance(center, radius);

      return distance.value(p);
    }

  private:
    const double radius = 0.1;
  };

  template <int dim>
  class SimulationCutMovingCylinder final : public Flow::CompressibleFlowCase<dim>
  {
  public:
    SimulationCutMovingCylinder(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Flow::CompressibleFlowCase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      Point<dim> lower_left;
      for (unsigned int d = 1; d < dim; ++d)
        lower_left[d] = 0.;

      Point<dim> upper_right;
      upper_right[0] = 1.0;
      for (unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 0.4;

      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 10;
      for (unsigned int d = 1; d < dim; ++d)
        subdivisions[d] = 4;

      GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                subdivisions,
                                                lower_left,
                                                upper_right);

      set_fitted_boundary_id(*this->triangulation);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      // fitted boundaries
      auto dummy_solution = std::make_shared<InitialFlowField<dim>>();
      this->attach_boundary_condition({0, dummy_solution}, "no_slip_wall", "compressible_flow");
    }

    void
    set_field_conditions() override
    {
      auto initial_condition = std::make_shared<InitialFlowField<dim>>();
      this->attach_initial_condition(initial_condition, "compressible_flow");

      const auto level_set =
        std::make_shared<MovingLevelSet<dim>>(this->parameters.time_stepping.start_time);
      this->attach_field_function(level_set, "level_set", "compressible_flow");

      // set velocity function of immersed object
      const auto unfitted_object_velocity = std::make_shared<UnfittedObjectVelocity<dim>>();
      this->attach_field_function(unfitted_object_velocity,
                                  "unfitted_object_velocity",
                                  "compressible_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim, double> &generic_data_out) const override
    {
      InitialFlowField<dim> reference_values;
      this->print_relative_norm(generic_data_out, reference_values, "norm");
    }

  private:
    /**
     * Set boundary id's for fitted boundaries
     */
    void
    set_fitted_boundary_id(const auto &triangulation) const
    {
      for (const auto &cell : triangulation.cell_iterators())
        if (cell->at_boundary())
          {
            for (const auto &face : cell->face_iterators())
              {
                const auto center = face->center();
                // adiabatic no-slip wall boundary conditions
                if ((std::fabs(center(0)) < 1e-12) or (std::fabs(center(0)) > 1. - 1e-12) or
                    (std::fabs(center(1)) < 1e-12) or (std::fabs(center(1)) > 0.4 - 1e-12))
                  face->set_boundary_id(0);
                else if (dim == 3 &&
                         ((std::fabs(center(2)) < 1e-12) or (std::fabs(center(2)) > 0.4 - 1e-12)))
                  face->set_boundary_id(0);
              }
          }
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
