#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/better_enum.hpp>

#include <memory>

#include "../dem_case.hpp"

namespace MeltPoolDG::Simulation::Dem
{
  template <int dim, typename number>
  class SimulationGenericHyperRectangleDomain final : public DemCase<dim, number>
  {
  public:
    SimulationGenericHyperRectangleDomain(std::string    parameter_file,
                                          const MPI_Comm mpi_communicator)
      : DemCase<dim, number>(parameter_file, mpi_communicator)
    {
      AssertThrow(dim > 1, dealii::ExcMessage("DEM simulations are only supported for dim > 1"));
    }

    void
    create_spatial_discretization() override
    {
      auto create_point_from_container =
        []<typename T>(const T &container) -> dealii::Point<dim, number> {
        dealii::Point<dim, number> point;
        if constexpr (dim == 2)
          point = dealii::Point<dim, number>(container[0], container[1]);
        else if constexpr (dim == 3)
          point = dealii::Point<dim, number>(container[0], container[1], container[2]);
        return point;
      };

      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim, number> dimensions = create_point_from_container(domain_dimensions);

      // TODO: How many cells are required
      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation,
        std::vector<unsigned int>(dim, 2*dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator)),
        dealii::Point<dim, number>(),
        dimensions,
        true);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * Reads boundary conditions, initial conditions, and domain size parameters from
     * the user input file.
     */
    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("case setup");
      {
        prm.add_parameter(
          "domain size",
          domain_dimensions,
          "Physical dimensions of the computational domain. "
          "Specify comma-separated values: the first for the x-direction, the second for the y-direction, "
          "and optionally the third for the z-direction.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Sizes of the domain in each dimension. The x, y, and z sizes are given at indices 0, 1,
    /// and 2, respectively.
    std::vector<number> domain_dimensions;
  };
} // namespace MeltPoolDG::Simulation::Dem
