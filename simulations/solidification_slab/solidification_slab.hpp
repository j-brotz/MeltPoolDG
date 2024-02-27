#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>


/**
 * This simulation represents a simple test example for heat transfer with solidification.
 * "A pseudo one-dimensional slab (material properties of ice / water) with length L = 1m is subject
 * to a fixed temperature Tˆ = 253K on its left edge at x = 0. The initial temperature in the whole
 * slab is T0 = 283K." [1]
 *
 * [1] Proell, S. D., Wall, W. A., & Meier, C. (2019). On phase change and latent heat models in
 * metal additive manufacturing process simulation. Advanced Modeling and Simulation in Engineering
 * Sciences, 7(1), 1--32. http://arxiv.org/abs/1906.06238
 *
 * Comini, G., Del Guidice, S., Lewis, R. W., & Zienkiewicz, O. C. (1974). Finite element solution
 * of non-linear heat conduction problems with special reference to phase change. International
 * Journal for Numerical Methods in Engineering, 8(3), 613–624.
 * https://doi.org/10.1002/nme.1620080314
 */

namespace MeltPoolDG::Simulation::SolidificationSlab
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static constexpr double L     = 1.0;
  static constexpr double T_0   = 283.0;
  static constexpr double T_hat = 253.0;

  template <int dim>
  class SimulationSolidificationSlab : public SimulationBase<dim>
  {
  public:
    SimulationSolidificationSlab(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
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

    void
    set_boundary_conditions() final
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

      this->attach_dirichlet_boundary_condition(
        left_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_hat), "heat_transfer");
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(T_0),
                                     "heat_transfer");
    }
  };
} // namespace MeltPoolDG::Simulation::SolidificationSlab
