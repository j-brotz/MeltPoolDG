#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include "../compressible_flow_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  template <int dim>
  class InflowFlowField : public Function<dim>
  {
  public:
    explicit InflowFlowField(const double time)
      : Function<dim>(dim + 2, time)
    {}

    double
    value(const Point<dim> &, const unsigned int component) const final
    {
      if (component == 0)
        return 1.;
      else if (component == 1)
        return 0.4;
      else if (component == dim + 1)
        return 3.097857142857143;
      else
        return 0.;
    }
  };

  template <int dim>
  class SimulationChannelParticleFlow final : public Flow::CompressibleFlowCase<dim>
  {
  public:
    SimulationChannelParticleFlow(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Flow::CompressibleFlowCase<dim>(parameter_file, mpi_communicator)
    {
      AssertThrow(
        dim > 1,
        dealii::ExcMessage(
          "The cylinder in channel flow case requires dim > 1 but the dimension is set to dim = " +
          std::to_string(dim) + "."));
    }

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      GridGenerator::channel_with_cylinder(*this->triangulation, 0.03, 1, 0, true);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      auto inflow_outflow_solution =
        std::make_shared<InflowFlowField<dim>>(this->parameters.time_stepping.start_time);
      auto dummy_solution =
        std::make_shared<InflowFlowField<dim>>(this->parameters.time_stepping.start_time);
      this->attach_boundary_condition({0, inflow_outflow_solution}, "inflow", "compressible_flow");
      this->attach_boundary_condition({1, inflow_outflow_solution},
                                      "outflow_fixed_energy",
                                      "compressible_flow");
      // particle surface
      this->attach_boundary_condition({2, dummy_solution}, "no_slip_wall", "compressible_flow");
      // channel boundary
      this->attach_boundary_condition({3, dummy_solution}, "slip_wall", "compressible_flow");
    }

    void
    set_field_conditions() override
    {
      auto initial_condition =
        std::make_shared<InflowFlowField<dim>>(this->parameters.time_stepping.start_time);
      this->attach_initial_condition(initial_condition, "compressible_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const override
    {
      InflowFlowField<dim> reference_values(generic_data_out.get_time());
      this->print_relative_norm(generic_data_out, reference_values, "Norm");
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
