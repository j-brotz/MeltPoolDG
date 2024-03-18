#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/post_processing/divergence_calc.hpp>
#include <meltpooldg/utilities/journal.hpp>

#include <iostream>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace EvaporatingDroplet
    {
      using namespace dealii;

      static double droplet_radius = 0.5;
      static double domain_length  = 4;
      static double bc_pressure    = 0.0;
      static double droplet_phi    = 1.0;

      BETTER_ENUM(DomainType, char, rectangle, ball)

      static DomainType domain_type = DomainType::rectangle;

      template <int dim>
      class InitialValuesLS : public Function<dim>
      {
      public:
        InitialValuesLS()
          : Function<dim>()
          , distance_sphere(Point<dim>(), droplet_radius)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
          return -droplet_phi * distance_sphere.value(p);
        }

      private:
        const Functions::SignedDistance::Sphere<dim> distance_sphere;
      };

      template <int dim>
      class SimulationEvaporatingDroplet : public SimulationBase<dim>
      {
      private:
        mutable std::shared_ptr<PostProcessingTools::DivergenceCalculator<dim>> diver;

      public:
        SimulationEvaporatingDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {
          AssertThrow(std::abs(droplet_phi) - 1.0 < 1e-10,
                      ExcMessage("'droplet phi' must be either 1 or"
                                 "-1."));
        }

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
            prm.add_parameter("bc pressure",
                              bc_pressure,
                              "Set the normal stress component on the outflow boundary.");
            prm.add_parameter("droplet phi",
                              droplet_phi,
                              "Set the level set value inside the droplet, either -1 or 1.");
          }
          prm.leave_subsection();
        }

        void
        create_spatial_discretization() override
        {
          // create triangulation
          if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
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
                5 * (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP ?
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

              if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
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

          if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
            this->triangulation->refine_global(this->parameters.base.global_refinements);
        }

        void
        set_boundary_conditions() override
        {
          // prescribe pressure dirichlet BC
          this->attach_open_boundary_condition(
            0, std::make_shared<Functions::ConstantFunction<dim>>(bc_pressure), "navier_stokes_u");
        }

        void
        set_field_conditions() override
        {
          this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(),
                                         "signed_distance");
          this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                           new Functions::ZeroFunction<dim>(dim)),
                                         "navier_stokes_u");
        }

        void
        do_postprocessing(const GenericDataOut<dim> &generic_data_out) const final
        {
          dealii::ConditionalOStream pcout(
            std::cout, Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
          if (!diver)
            diver = std::make_shared<PostProcessingTools::DivergenceCalculator<dim>>(
              generic_data_out, "interface_velocity");
          else
            // @todo: We need to reinit, since generic_data_out is currently created
            // for every time step.
            diver->reinit(generic_data_out);


          diver->process(0 /*does not matter*/);
          std::ostringstream str;
          str << "∇·uΓ = " << std::setprecision(10) << std::scientific << diver->get_divergence();

          Journal::print_line(pcout, str.str(), "user_defined_postprocess", 4);
          ;
        }
      };
    } // namespace EvaporatingDroplet
  }   // namespace Simulation
} // namespace MeltPoolDG
