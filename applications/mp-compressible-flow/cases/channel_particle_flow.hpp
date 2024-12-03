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
    {}

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
      this->attach_boundary_condition({2, dummy_solution}, "no_slip_wall", "compressible_flow");
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

  private:
    // for self-registration
    static SimulationCaseRegistrar<Flow::CompressibleFlowCase<dim>> registrar;
  };

  // for self-registration
  template <int dim>
  SimulationCaseRegistrar<Flow::CompressibleFlowCase<dim>>
    SimulationChannelParticleFlow<dim>::registrar(
      "channel_particle_flow",
      [](const std::string &parameter_file, const MPI_Comm mpi_communicator) {
        return std::make_unique<SimulationChannelParticleFlow<dim>>(parameter_file,
                                                                    mpi_communicator);
      });
} // namespace MeltPoolDG::Simulation::CompressibleFlow
