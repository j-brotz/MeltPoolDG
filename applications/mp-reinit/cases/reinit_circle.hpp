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

#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/utilities/characteristic_functions.hpp>

#include <cmath>

#include "../reinitialization_case.hpp"

namespace MeltPoolDG::Simulation::ReinitCircle
{
  template <int dim, typename number>
  class InitializePhi : public dealii::Function<dim, number>
  {
  public:
    InitializePhi()
      : dealii::Function<dim, number>()
      , distance_sphere(dim == 1 ? dealii::Point<dim, number>(0.0) :
                                   dealii::Point<dim, number>(0.0, 0.5),
                        0.25)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return CharacteristicFunctions::sgn(-distance_sphere.value(p));
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
  };

  template <int dim, typename number>
  class ExactSolution : public dealii::Function<dim, number>
  {
  public:
    ExactSolution(const number eps)
      : dealii::Function<dim, number>()
      , distance_sphere(dealii::Point<dim, number>(0.0, 0.5), 0.25)
      , eps_interface(eps)
    {}

    number
    value(const dealii::Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return CharacteristicFunctions::tanh_characteristic_function(-distance_sphere.value(p),
                                                                   eps_interface);
    }

  private:
    const dealii::Functions::SignedDistance::Sphere<dim> distance_sphere;
    const number                                         eps_interface;
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim, typename number>
  class SimulationReinit : public LevelSet::ReinitializationCase<dim, number>
  {
  public:
    SimulationReinit(std::string parameter_file, const MPI_Comm mpi_communicator)
      : LevelSet::ReinitializationCase<dim, number>(parameter_file, mpi_communicator)
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
    {}

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<InitializePhi<dim, number>>(), "level_set");
    }

  private:
    number left_domain  = -1.0;
    number right_domain = 1.0;
  };
} // namespace MeltPoolDG::Simulation::ReinitCircle
