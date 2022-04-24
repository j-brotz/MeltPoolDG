#pragma once

// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <iostream>

// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace EvaporatingDroplet
    {
      using namespace dealii;

      static double droplet_radius = 0.5;
      static double domain_length  = 4;

      BETTER_ENUM(DomainType, char, rectangle, ball)

      static DomainType domain_type = DomainType::rectangle;

      template <int dim>
      class InitialValuesLS : public Function<dim>
      {
      public:
        InitialValuesLS(const double eps)
          : Function<dim>()
          , distance_sphere(Point<dim>(), droplet_radius)
          , eps(eps)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const
        {
          return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
            -distance_sphere.value(p), eps);
        }

      private:
        const Functions::SignedDistance::Sphere<dim> distance_sphere;
        const double                                 eps;
      };

      template <int dim>
      class SimulationEvaporatingDroplet : public SimulationBase<dim>
      {
      public:
        SimulationEvaporatingDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {}

        void
        add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
        {
          prm.enter_subsection("simulation specific");
          {
            prm.add_parameter("droplet radius", droplet_radius, "Set the radius of the droplet.");
            prm.add_parameter("domain length",
                              domain_length,
                              "Set the characteristic length of the domain.");
            prm.add_parameter("domain type", domain_type, "Set the type of the domain.");
          }
          prm.leave_subsection();
        }

        void
        create_spatial_discretization() override
        {
          // create triangulation
          if (this->parameters.base.do_simplex || dim == 1)
            this->triangulation =
              std::make_shared<parallel::shared::Triangulation<dim>>(this->mpi_communicator);
          else
            this->triangulation =
              std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

          const double half_length = domain_length / 2.;

          AssertThrow(droplet_radius < half_length,
                      ExcMessage(
                        "The droplet radius exceeds the domain size. "
                        "You might adjust either the domain length or the droplet radius."));

          if (domain_type == DomainType::rectangle)
            {
              std::vector<unsigned int> subdivisions(
                dim,
                5 * (this->parameters.base.do_simplex ?
                       Utilities::pow(2, this->parameters.base.global_refinements) :
                       1));

              const Point<dim> bottom_left =
                (dim == 1 ? Point<dim>(-half_length) :
                 dim == 2 ? Point<dim>(-half_length, -half_length) :
                            Point<dim>(-half_length, -half_length, -half_length));
              const Point<dim> top_right =
                (dim == 1 ? Point<dim>(half_length) :
                 dim == 2 ? Point<dim>(half_length, half_length) :
                            Point<dim>(half_length, half_length, half_length));

              if (this->parameters.base.do_simplex)
                GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                         subdivisions,
                                                                         bottom_left,
                                                                         top_right);
              else
                GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                          subdivisions,
                                                          bottom_left,
                                                          top_right);
            }
          else if (domain_type == DomainType::ball)
            {
              // create circular domain
              const Point<dim> center;
              GridGenerator::hyper_ball(*this->triangulation, center, half_length /*radius*/);
            }
          else
            AssertThrow(false, ExcNotImplemented());

          if (this->parameters.base.do_simplex == false)
            this->triangulation->refine_global(this->parameters.base.global_refinements);
        }

        void
        set_boundary_conditions() override
        {
          // zero normal stress component
          this->attach_open_boundary_condition(0, "navier_stokes_u");
        }

        void
        set_field_conditions() override
        {
          double eps =
            UtilityFunctions::compute_initial_epsilon<dim>(this->parameters, *this->triangulation);

          AssertThrow(eps > 0, ExcNotImplemented());

          this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps), "level_set");
          this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                           new Functions::ZeroFunction<dim>(dim)),
                                         "navier_stokes_u");
        }
      };

    } // namespace EvaporatingDroplet
  }   // namespace Simulation
} // namespace MeltPoolDG
