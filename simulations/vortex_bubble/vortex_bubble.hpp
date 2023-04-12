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
    namespace VortexBubble
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
          , distance_sphere(dim == 1 ? Point<dim>(0.5) : Point<dim>(0.5, 0.75), 0.15)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
          return -distance_sphere.value(p);
        }

      private:
        const Functions::SignedDistance::Sphere<dim> distance_sphere;
      };

      template <int dim>
      class AdvectionField : public Function<dim>
      {
      public:
        AdvectionField()
          : Function<dim>(dim)
        {}

        void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
          if constexpr (dim == 2)
            {
              const double time = this->get_time();

              const double Tf = 2.0;
              const double x  = p[0];
              const double y  = p[1];

              const double reverseCoefficient = std::cos(numbers::PI * time / Tf);

              values[0] = reverseCoefficient * (std::sin(2. * numbers::PI * y) *
                                                std::pow(std::sin(numbers::PI * x), 2.));
              values[1] = reverseCoefficient * (-std::sin(2. * numbers::PI * x) *
                                                std::pow(std::sin(numbers::PI * y), 2.));
            }
          else
            AssertThrow(false, ExcMessage("Advection field for dim!=2 not implemented"));
        }
      };

      /* for constant Dirichlet conditions we could also use the ConstantFunction
       * utility from dealii
       */
      template <int dim>
      class DirichletCondition : public Function<dim>
      {
      public:
        DirichletCondition()
          : Function<dim>()
        {}

        double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
          (void)p;
          (void)component;
          return -1.0;
        }
      };

      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationVortexBubble : public SimulationBase<dim>
      {
      public:
        SimulationVortexBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {}

        void
        create_spatial_discretization() override
        {
          if (dim == 1 || this->parameters.base.do_simplex)
            {
              AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
              this->triangulation = std::make_shared<Triangulation<dim>>();
            }
          else
            {
              this->triangulation =
                std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
            }

          if (this->parameters.base.do_simplex)
            {
              GridGenerator::subdivided_hyper_cube_with_simplices(
                *this->triangulation,
                Utilities::pow(2, this->parameters.base.global_refinements),
                left_domain,
                right_domain);
            }
          else
            {
              GridGenerator::subdivided_hyper_cube(*this->triangulation,
                                                   2,
                                                   left_domain,
                                                   right_domain);
              this->triangulation->refine_global(this->parameters.base.global_refinements - 1);
            }
        }

        void
        set_boundary_conditions() override
        {
          // none
        }

        void
        set_field_conditions() override
        {
          this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(), "signed_distance");
          this->attach_advection_field(std::make_shared<AdvectionField<dim>>(), "level_set");
        }

      private:
        double left_domain  = 0.0;
        double right_domain = 1.0;
      };

    } // namespace VortexBubble
  }   // namespace Simulation
} // namespace MeltPoolDG
