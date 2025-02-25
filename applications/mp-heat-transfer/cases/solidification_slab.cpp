#include "solidification_slab.hpp"
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/case_registration.hpp>

#include <cmath>
#include <vector>


namespace MeltPoolDG::Simulation::SolidificationSlab
{
  static constexpr double L     = 1.0;
  static constexpr double T_0   = 283.0;
  static constexpr double T_hat = 253.0;


  template <int dim>
  SimulationSolidificationSlab<dim>::SimulationSolidificationSlab(std::string    parameter_file,
                                                                  const MPI_Comm mpi_communicator)
    : Heat::HeatTransferCase<dim>(parameter_file, mpi_communicator)
  {}


  template <int dim>
  void
  SimulationSolidificationSlab<dim>::create_spatial_discretization()
  {
    if constexpr (dim == 1)
      {
        AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
        this->triangulation = std::make_shared<Triangulation<dim>>();
        // create mesh
        const Point<1> left(0);
        const Point<1> right(L);
        GridGenerator::hyper_rectangle(*this->triangulation, left, right);
        this->triangulation->refine_global(this->parameters.base.global_refinements);
      }
    else if constexpr (dim == 2)
      {
        this->triangulation =
          std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        const unsigned int        num_el = std::pow(2, this->parameters.base.global_refinements);
        std::vector<unsigned int> refinements(dim, 1);
        refinements[0] = num_el;
        // create mesh
        const Point<2> left(0, 0);
        const Point<2> right(L, 1. / num_el);
        GridGenerator::subdivided_hyper_rectangle(*this->triangulation, refinements, left, right);
      }
    else if constexpr (dim == 3)
      {
        this->triangulation =
          std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        const int                 num_el = std::pow(2, this->parameters.base.global_refinements);
        std::vector<unsigned int> refinements(dim, 1);
        refinements[0] = num_el;
        // create mesh
        const Point<3> left(0, 0, 0);
        const Point<3> right(L, 1. / num_el, 1. / num_el);
        GridGenerator::subdivided_hyper_rectangle(*this->triangulation, refinements, left, right);
      }
    else
      AssertThrow(false, ExcMessage("Impossible dimension! Abort ..."));
  }


  template <int dim>
  void
  SimulationSolidificationSlab<dim>::set_boundary_conditions()
  {
    /*
     *  create a pair of (boundary_id, dirichlet_function)
     */

    const types::boundary_id left_bc = 10;

    for (const auto &cell : this->triangulation->cell_iterators())
      for (auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            if (face->center()[0] == 0.0)
              face->set_boundary_id(left_bc);
          }

    this->attach_boundary_condition({left_bc,
                                     std::make_shared<Functions::ConstantFunction<dim>>(T_hat)},
                                    "dirichlet",
                                    "heat_transfer");
  }


  template <int dim>
  void
  SimulationSolidificationSlab<dim>::set_field_conditions()
  {
    this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(T_0),
                                   "heat_transfer");
  }

  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase,
                           SimulationSolidificationSlab,
                           "solidification_slab",
                           1);
  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase,
                           SimulationSolidificationSlab,
                           "solidification_slab",
                           2);
  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase,
                           SimulationSolidificationSlab,
                           "solidification_slab",
                           3);
} // namespace MeltPoolDG::Simulation::SolidificationSlab
