#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/post_processing/generic_data_out.hpp>

#include <string>

#include "../meltpool_vapor_flow_case.hpp"

namespace MeltPoolDG::Simulation::MeltPoolVaporFlow
{
  template <int dim, typename number>
  class SimulationHyperRectangleBox final : public MeltPoolDG::MeltPoolVaporFlow::Case<dim, number>
  {
  public:
    SimulationHyperRectangleBox(std::string parameter_file, const MPI_Comm mpi_communicator)
      : MeltPoolDG::MeltPoolVaporFlow::Case<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      // TODO: Use dealii::GridGenerator::subdivided_hyper_rectangle to create the triangulation.
    }

    void
    set_boundary_conditions() override
    {
      // TODO: Set the boundary conditions for both problems (meltpool and compressible vapor flow)
    }

    void
    set_field_conditions() override
    {
      // TODO: Set the initial conditions for both problems (meltpool and compressible vapor flow)
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      // TODO: Case specific postprocessing, whatever is necessary for this case.
    }

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      // TODO: In case there are any simulation specific parameters, add them here.
    }
  };
} // namespace MeltPoolDG::Simulation::MeltPoolVaporFlow
