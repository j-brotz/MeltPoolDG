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
  class SteadyInflowField : public Function<dim>
  {
  public:
    explicit SteadyInflowField()
      : Function<dim>(dim + 2)
    {
      Assert(dim == 2 or dim == 3, ExcNotImplemented());
    }

    double
    value(const Point<dim> &, const unsigned int component) const final
    {
      if (component == 0)
        return 0.001;
      else if (component == 1)
        return 0.01;
      else if (component == dim + 1)
        return 194.5142;
      else
        return 0.;
    }
  };

  template <int dim>
  class SimulationCutUnfittedInflow final : public Flow::CompressibleFlowCase<dim>
  {
  public:
    SimulationCutUnfittedInflow(std::string parameter_file, const MPI_Comm mpi_communicator)
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
        upper_right[d] = 0.008;

      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 125;
      for (unsigned int d = 1; d < dim; ++d)
        subdivisions[d] = 1;

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
      auto inflow_outflow_solution = std::make_shared<SteadyInflowField<dim>>();
      auto dummy_solution          = std::make_shared<SteadyInflowField<dim>>();
      this->attach_boundary_condition({BoundaryID::subsonic_outflow_with_fixed_energy,
                                       inflow_outflow_solution},
                                      "outflow_fixed_energy",
                                      "compressible_flow");
      this->attach_boundary_condition({BoundaryID::slip_wall, dummy_solution},
                                      "slip_wall",
                                      "compressible_flow");
    }

    void
    set_field_conditions() override
    {
      auto initial_condition = std::make_shared<SteadyInflowField<dim>>();
      this->attach_initial_condition(initial_condition, "compressible_flow");

      // set level-set function
      Point<dim> center;
      center[0] = 0.5;
      for (unsigned int d = 1; d < dim; ++d)
        center[d] = 0.004;

      dealii::Tensor<1, dim> normal;
      normal[0] = 1.;

      const auto level_set =
        std::make_shared<Functions::SignedDistance::Plane<dim>>(center, normal);
      this->attach_field_function(level_set, "level_set", "compressible_flow");

      // set inflow function at unfitted object
      this->attach_field_function(initial_condition, "unfitted_inflow", "compressible_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim, double> &generic_data_out) const override
    {
      SteadyInflowField<dim> reference_values;
      this->print_relative_norm(generic_data_out, reference_values, "norm");
    }

  private:
    /**
     * Enumeration for the boundary id's.
     */
    enum BoundaryID
    {
      subsonic_outflow_with_fixed_energy,
      slip_wall
    };

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
                // outflow boundary with fixed energy
                if ((std::fabs(center(0)) > 1. - 1e-12))
                  face->set_boundary_id(BoundaryID::subsonic_outflow_with_fixed_energy);
                // slip wall boundary conditions
                else if ((std::fabs(center(1)) < 1e-12) or (std::fabs(center(1)) > 0.008 - 1e-12))
                  face->set_boundary_id(BoundaryID::slip_wall);
              }
          }
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
