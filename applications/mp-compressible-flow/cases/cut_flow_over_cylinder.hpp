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
  class FlowField : public Function<dim>
  {
  public:
    explicit FlowField()
      : Function<dim>(dim + 2)
    {}

    double
    value(const Point<dim> &, const unsigned int component) const final
    {
      if (component == 0)
        return 1.;
      else if (component == 1)
        return 0.1;
      else if (component == dim + 1)
        return 3.0228571429;
      else
        return 0.;
    }
  };

  template <int dim>
  class SimulationCutFlowOverCylinder final : public Flow::CompressibleFlowCase<dim>
  {
  public:
    SimulationCutFlowOverCylinder(std::string parameter_file, const MPI_Comm mpi_communicator)
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
      auto inflow_outflow_solution = std::make_shared<FlowField<dim>>();
      auto dummy_solution          = std::make_shared<FlowField<dim>>();
      this->attach_boundary_condition({BoundaryID::inflow, inflow_outflow_solution},
                                      "inflow",
                                      "compressible_flow");
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
      auto initial_condition = std::make_shared<FlowField<dim>>();
      this->attach_initial_condition(initial_condition, "compressible_flow");

      // set level-set function
      Point<dim> center;
      center[0] = 0.3;
      center[1] = 0.2;
      if constexpr (dim == 3)
        center[2] = 0.2;

      const double radius = 0.1;

      const auto level_set =
        std::make_shared<Functions::SignedDistance::Sphere<dim>>(center, radius);
      this->attach_field_function(level_set, "level_set", "compressible_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const override
    {
      FlowField<dim> reference_values;
      this->print_relative_norm(generic_data_out, reference_values, "Norm");
    }

  private:
    // for self-registration
    static SimulationCaseRegistrar<Flow::CompressibleFlowCase<dim>> registrar;

    /**
     * Enumeration for the fitted boundary id's.
     */
    enum BoundaryID
    {
      inflow,
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
                // inflow boundary
                if ((std::fabs(center(0)) < 1e-12))
                  face->set_boundary_id(BoundaryID::inflow);
                // outflow boundary with fixed energy
                else if ((std::fabs(center(0)) > 1. - 1e-12))
                  face->set_boundary_id(BoundaryID::subsonic_outflow_with_fixed_energy);
                // slip wall boundary conditions
                else if ((std::fabs(center(1)) < 1e-12) or (std::fabs(center(1)) > 0.4 - 1e-12))
                  face->set_boundary_id(BoundaryID::slip_wall);
                else if (dim == 3 and
                         ((std::fabs(center(2)) < 1e-12) or (std::fabs(center(2)) > 0.4 - 1e-12)))
                  face->set_boundary_id(BoundaryID::slip_wall);
              }
          }
    }
  };

  // for self-registration
  template <int dim>
  SimulationCaseRegistrar<Flow::CompressibleFlowCase<dim>>
    SimulationCutFlowOverCylinder<dim>::registrar(
      "cut_flow_over_cylinder",
      [](const std::string &parameter_file, const MPI_Comm mpi_communicator) {
        return std::make_unique<SimulationCutFlowOverCylinder<dim>>(parameter_file,
                                                                    mpi_communicator);
      });
} // namespace MeltPoolDG::Simulation::CompressibleFlow
