#pragma once
// MeltPoolDG
#include <deal.II/base/mpi.h>

#include <meltpooldg/core/base_data.hpp>
#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/simulation_base.hpp>

#include <memory>
#include <string>

namespace MeltPoolDG
{
  namespace Simulation
  {
    template <int dim, typename number>
    class SimulationSelector
    {
    public:
      static std::unique_ptr<MeltPoolCase<dim, number>>
      get_simulation(const std::string simulation_name,
                     const std::string parameter_file,
                     const MPI_Comm    mpi_communicator);
    };
  } // namespace Simulation
} // namespace MeltPoolDG
