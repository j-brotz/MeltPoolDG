#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include "meltpooldg/particles/particle.hpp"
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/utilities/better_enum.hpp>

#include <memory>

#include "../dem_case.hpp"

namespace MeltPoolDG::Simulation::Dem
{
  template <int dim, typename number>
  class SignedDistancePlane : public dealii::Function<dim, number>
  {
  }

  template <int dim, typename number>
  class ParticlesInBox final : public DemCase<dim, number>
  {
  public:
    ParticlesInBox(std::string parameter_file, const MPI_Comm mpi_communicator)
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

      ObstacleTriangulationDataStructure<dim, number, SphericalParticle<dim, number>>::
        setup_hyper_rectangular_triangulation(*this->triangulation,
                                              std::make_pair(dealii::Point<dim, number>(),
                                                             dimensions),
                                              max_particle_influence_radius); // TODO
    }

    /**
     * Dummy function since boundary conditions are not needed for DEM simulations. The function is
     * required to be implemented since it is a pure virtual function in the base class.
     */
    void
    set_boundary_conditions() override
    {}

    /**
     * Dummy function since inital conditions are directly read from the particle input file. The
     * function is required to be implemented since it is a pure virtual function in the base class.
     */
    void
    set_field_conditions() override
    {}

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

        prm.add_parameter(
          "max particle influence radius",
          max_particle_influence_radius,
          "Maximum influence radius of particles, used for determining the necessary mesh resolution.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Sizes of the domain in each dimension. The x, y, and z sizes are given at indices 0, 1,
    /// and 2, respectively.
    std::vector<number> domain_dimensions;

    number max_particle_influence_radius = 60e-6; // TODO
  };
} // namespace MeltPoolDG::Simulation::Dem
