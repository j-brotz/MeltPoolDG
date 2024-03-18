#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace ReinitCircle
    {
      using namespace dealii;
      /*
       * this function specifies the initial field of the level set equation
       */

      template <int dim>
      class InitializePhi : public Function<dim>
      {
      public:
        InitializePhi()
          : Function<dim>()
          , distance_sphere(dim == 1 ? Point<dim>(0.0) : Point<dim>(0.0, 0.5), 0.25)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
          return UtilityFunctions::CharacteristicFunctions::sgn(-distance_sphere.value(p));
        }

      private:
        const Functions::SignedDistance::Sphere<dim> distance_sphere;
      };

      template <int dim>
      class ExactSolution : public Function<dim>
      {
      public:
        ExactSolution(const double eps)
          : Function<dim>()
          , distance_sphere(Point<dim>(0.0, 0.5), 0.25)
          , eps_interface(eps)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
          return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
            -distance_sphere.value(p), eps_interface);
        }

      private:
        const Functions::SignedDistance::Sphere<dim> distance_sphere;
        const double                                 eps_interface;
      };
      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationReinit : public SimulationBase<dim>
      {
      public:
        SimulationReinit(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {}

        void
        create_spatial_discretization() override
        {
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
          this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(), "level_set");
          this->attach_exact_solution(std::make_shared<ExactSolution<dim>>(0.01), "level_set");
        }

      private:
        double left_domain  = -1.0;
        double right_domain = 1.0;
      };

    } // namespace ReinitCircle
  }   // namespace Simulation
} // namespace MeltPoolDG
