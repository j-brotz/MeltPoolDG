#pragma once
// MeltPoolDG
#include <deal.II/base/mpi.h>

#include <meltpooldg/interface/base_data.hpp>
#include <meltpooldg/interface/simulation_base.hpp>

#include <memory>
#include <string>

namespace MeltPoolDG
{
  namespace Simulation
  {
    template <int dim>
    class SimulationSelector
    {
    public:
      static std::shared_ptr<SimulationBase<dim>>
      get_simulation(const ApplicationName simulation_name,
                     const std::string     parameter_file,
                     const MPI_Comm        mpi_communicator);
    };
  } // namespace Simulation
} // namespace MeltPoolDG
