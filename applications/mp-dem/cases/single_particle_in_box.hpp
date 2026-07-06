#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

// #include "meltpooldg/particles/particle.hpp"
#include <meltpooldg/particles/obstacle_data_structure.hpp>
#include <meltpooldg/utilities/better_enum.hpp>

#include <memory>

#include "../dem_case.hpp"

namespace MeltPoolDG::Simulation::Dem
{
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
      this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
        this->mpi_communicator,
        dealii::Triangulation<dim>::MeshSmoothing::none,
        dealii::parallel::distributed::Triangulation<dim>::Settings::construct_multigrid_hierarchy);
      dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        n_subdivisions_per_dimension,
                                                        bounding_box_corners.first,
                                                        bounding_box_corners.second);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * Dummy function since boundary conditions are not needed for DEM simulations. The function is
     * required to be implemented since it is a pure virtual function in the base class.
     */
    void
    set_boundary_conditions() override
    {
      if constexpr (dim == 2)
        {
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.first, dealii::Point<dim>(1, 0)));
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.first, dealii::Point<dim>(0, 1)));
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.second, dealii::Point<dim>(-1, 0)));
        }

      if constexpr (dim == 3)
        {
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.first, dealii::Point<dim>(1, 0, 0)));
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.first, dealii::Point<dim>(0, 1, 0)));
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.first, dealii::Point<dim>(0, 0, 1)));
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.second, dealii::Point<dim>(-1, 0, 0)));
          this->walls.push_back(std::make_unique<dealii::Functions::SignedDistance::Plane<dim>>(
            bounding_box_corners.second, dealii::Point<dim>(0, -1, 0)));
        }
    }

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
    add_case_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.add_parameter(
        "bounding box corner 1",
        bounding_box_corners.first,
        "Phyiscal location of the first corner of the bounding box containing all particles (e.g. the lower left corner). The x, y, and z "
        "coordinates are given at indices 0, 1, and 2, respectively.");

      prm.add_parameter(
        "bounding box corner 2",
        bounding_box_corners.second,
        "Phyiscal location of the second corner of the bounding box containing all particles (e.g. the upper right corner). The x, y, and z "
        "coordinates are given at indices 0, 1, and 2, respectively.");

      prm.add_parameter(
        "max particle influence radius",
        max_particle_influence_radius,
        "Maximum influence radius of particles, used for determining the necessary mesh resolution.");

      prm.add_parameter(
        "n subdivisions",
        n_subdivisions_per_dimension,
        "Number of subdivisions in each dimension for the initial mesh. If this parameter is not provided, the mesh will be refined uniformly until the cell size is smaller than the maximum particle influence radius which is the optimal case for the particle search data structure. ");

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Sizes of the domain in each dimension. The x, y, and z sizes are given at indices 0, 1,
    /// and 2, respectively.
    std::pair<dealii::Point<dim, number>, dealii::Point<dim, number>> bounding_box_corners;

    // std::array<std::unique_ptr<dealii::Functions::SignedDistance::Plane<dim>>, 2 * dim - 1>
    // box_boundaries;

    std::vector<unsigned> n_subdivisions_per_dimension;

    number max_particle_influence_radius;
  };
} // namespace MeltPoolDG::Simulation::Dem
