#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>

#include "../level_set_case.hpp"

namespace MeltPoolDG::Simulation::RotatingBubble
{
  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim>
  {
  public:
    InitializePhi()
      : dealii::Function<dim>()
      , distance_sphere(dim == 1 ? dealii::Point<dim, number>(0.0) :
                        dim == 2 ? dealii::Point<dim, number>(0.0, 0.5) :
                                   dealii::Point<dim, number>(0.0, 0.5, 0.0),
                        0.25 /*radius*/)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return -distance_sphere.value(p);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim, typename number>
  class AdvectionField : public dealii::Function<dim, number>
  {
  public:
    AdvectionField()
      : dealii::Function<dim, number>(dim)
    {}

    void
    vector_value(const dealii::Point<dim, number> &p, dealii::Vector<number> &values) const override
    {
      if constexpr (dim >= 2)
        {
          const number x = p[0];
          const number y = p[1];

          values[0] = 4 * y;
          values[1] = -4 * x;
        }
      else
        AssertThrow(false, dealii::ExcMessage("Advection field for dim!=2 not implemented"));

      if constexpr (dim == 3)
        values[2] = 0;
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationRotatingBubble : public LevelSet::LevelSetCase<dim, number>
  {
  public:
    SimulationRotatingBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::LevelSetCase<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      using namespace dealii;
      if (dim == 1 || this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<Triangulation<dim>>();
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          GridGenerator::subdivided_hyper_cube_with_simplices(
            *this->triangulation,
            Utilities::pow(2, this->parameters.base.global_refinements),
            left_domain,
            right_domain);
        }
      else
        {
          GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
    }

    void
    set_boundary_conditions() override
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */
      constexpr dealii::types::boundary_id inflow_bc  = 42;
      constexpr dealii::types::boundary_id do_nothing = 0;

      this->attach_boundary_condition(
        {inflow_bc, std::make_shared<dealii::Functions::ConstantFunction<dim>>(-1.0)},
        "dirichlet",
        "level_set");
      /*
       *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
       *  due to the hyperbolic nature of the analyzed problem
       *
                  out    in
      (-1,1)  +---------------+ (1,1)
              |       :       |
        in    |       :       | out
              |_______________|
              |       :       |
        out   |       :       | in
              |       :       |
              +---------------+
       * (-1,-1)  in     out   (1,-1)
       */
      if constexpr (dim > 1)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  const number half_line = (right_domain + left_domain) / 2;

                  if (face->center()[0] == left_domain && face->center()[1] > half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[0] == right_domain && face->center()[1] < half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[1] == right_domain && face->center()[0] > half_line)
                    face->set_boundary_id(inflow_bc);
                  else if (face->center()[1] == left_domain && face->center()[0] < half_line)
                    face->set_boundary_id(inflow_bc);
                  else
                    face->set_boundary_id(do_nothing);
                }
        }
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(),
                                     "signed_distance");
      this->attach_field_function(std::make_shared<AdvectionField<dim, number>>(),
                                  "prescribed_velocity",
                                  "level_set");
    }

  private:
    number left_domain  = -1.0;
    number right_domain = 1.0;
  };
} // namespace MeltPoolDG::Simulation::RotatingBubble
