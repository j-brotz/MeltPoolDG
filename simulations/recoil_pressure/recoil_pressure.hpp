#pragma once

// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace RecoilPressure
    {
      const double T_initial = 500.;

      using namespace dealii;

      template <int dim>
      class InitialValuesLS : public Function<dim>
      {
      public:
        InitialValuesLS(const double x_min,
                        const double x_max,
                        const double y_min,
                        const double y_interface)
          : Function<dim>()
          , x_min(x_min)
          , x_max(x_max)
          , y_min(y_min)
          , y_interface(y_interface)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const
        {
          Point<dim> lower_left =
            dim == 2 ? Point<dim>(x_min, y_min) : Point<dim>(x_min, x_min, y_min);
          Point<dim> upper_right =
            dim == 2 ? Point<dim>(x_max, y_interface) : Point<dim>(x_max, x_max, y_interface);

          return UtilityFunctions::CharacteristicFunctions::sgn(
            DistanceFunctions::rectangular_manifold<dim>(p, lower_left, upper_right));
        }
        double x_min, x_max, y_min, y_interface;
      };

      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationRecoilPressure : public SimulationBase<dim>
      {
      private:
        double domain_x_min = 0;
        double domain_x_max = 0;
        double domain_y_min = 0;
        double domain_y_max = 0;

      public:
        SimulationRecoilPressure(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {}

        void
        add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
        {
          prm.enter_subsection("simulation specific domain");
          {
            prm.add_parameter("domain x min",
                              domain_x_min,
                              "minimum x coordinate of simulation domain");
            prm.add_parameter("domain y min",
                              domain_y_min,
                              "minimum y coordinate of simulation domain");
            prm.add_parameter("domain x max",
                              domain_x_max,
                              "maximum x coordinate of simulation domain");
            prm.add_parameter("domain y max",
                              domain_y_max,
                              "maximum y coordinate of simulation domain");
          }
          prm.leave_subsection();
        }

        void
        create_spatial_discretization() override
        {
          if (this->parameters.base.do_simplex)
            {
#ifdef DEAL_II_WITH_METIS
              this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
                this->mpi_communicator,
                (Triangulation<dim>::none),
                false,
                parallel::shared::Triangulation<dim>::Settings::partition_metis);
#else
              AssertThrow(
                false,
                ExcMessage(
                  "Missing Metis support of the deal.II installation. "
                  "Configure deal.II with -D DEAL_II_WITH_METIS='ON' to execute this example."));
#endif
            }
          else
            {
              this->triangulation =
                std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
            }

          const double &x_min = domain_x_min;
          const double &x_max = domain_x_max;
          const double &y_min = domain_y_min;
          const double &y_max = domain_y_max;

          if constexpr ((dim == 2) || (dim == 3))
            {
              // create mesh
              const Point<dim> bottom_left =
                (dim == 2) ? Point<dim>(x_min, y_min) : Point<dim>(x_min, x_min, y_min);
              const Point<dim> top_right =
                (dim == 2) ? Point<dim>(x_max, y_max) : Point<dim>(x_max, x_max, y_max);

              if (this->parameters.base.do_simplex)
                {
                  // create mesh
                  std::vector<unsigned int> subdivisions(
                    dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
                  subdivisions[dim - 1] *= 2;

                  GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                           subdivisions,
                                                                           bottom_left,
                                                                           top_right);
                }
              else
                {
                  GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
                  this->triangulation->refine_global(this->parameters.base.global_refinements);
                }
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }

        void
        set_boundary_conditions() override
        {
          if (this->parameters.mp.do_evaporation)
            {
              const types::boundary_id lower_bc = 1;
              const types::boundary_id upper_bc = 2;
              const types::boundary_id left_bc  = 3;
              const types::boundary_id right_bc = 4;

              if constexpr (dim == 2)
                {
                  for (const auto &cell : this->triangulation->cell_iterators())
                    for (const auto &face : cell->face_iterators())
                      if ((face->at_boundary()))
                        {
                          if (face->center()[1] == domain_y_min)
                            face->set_boundary_id(lower_bc);
                          else if (face->center()[1] == domain_y_max)
                            face->set_boundary_id(upper_bc);
                          else if (face->center()[0] == domain_x_min)
                            face->set_boundary_id(left_bc);
                          else if (face->center()[0] == domain_x_max)
                            face->set_boundary_id(right_bc);
                        }
                }
              else
                AssertThrow(false, ExcNotImplemented());

              this->attach_symmetry_boundary_condition(left_bc, "navier_stokes_u");
              this->attach_symmetry_boundary_condition(right_bc, "navier_stokes_u");
              this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
              this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");

              this->attach_dirichlet_boundary_condition(
                upper_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1.0), "level_set");
              /*
               * BC for heat transfer
               */
              if (this->parameters.laser.heat_source_model != "Analytical")
                {
                  this->attach_dirichlet_boundary_condition(
                    lower_bc,
                    std::make_shared<Functions::ConstantFunction<dim>>(T_initial),
                    "heat_transfer");
                  this->attach_dirichlet_boundary_condition(
                    upper_bc,
                    std::make_shared<Functions::ConstantFunction<dim>>(T_initial),
                    "heat_transfer");
                  this->attach_dirichlet_boundary_condition(
                    left_bc,
                    std::make_shared<Functions::ConstantFunction<dim>>(T_initial),
                    "heat_transfer");
                  this->attach_dirichlet_boundary_condition(
                    right_bc,
                    std::make_shared<Functions::ConstantFunction<dim>>(T_initial),
                    "heat_transfer");
                }
            }
          else if (this->parameters.base.problem_name == "melt_pool")
            {
              this->attach_no_slip_boundary_condition(0, "navier_stokes_u");
              this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");
              /*
               * BC for heat transfer
               */
              if (this->parameters.laser.heat_source_model != "Analytical")
                {
                  this->attach_dirichlet_boundary_condition(
                    0,
                    std::make_shared<Functions::ConstantFunction<dim>>(T_initial),
                    "heat_transfer");
                }
            }
          else
            AssertThrow(false, ExcNotImplemented());
        }

        void
        set_field_conditions() override
        {
          auto laser_center = MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
            this->parameters.laser.center);
          this->attach_initial_condition(
            std::make_shared<InitialValuesLS<dim>>(
              domain_x_min, domain_x_max, domain_y_min, laser_center[dim - 1]),
            "level_set");
          this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                           new Functions::ZeroFunction<dim>(dim)),
                                         "navier_stokes_u");
          if (this->parameters.laser.heat_source_model != "Analytical")
            this->attach_initial_condition(
              std::make_shared<Functions::ConstantFunction<dim>>(T_initial), "heat_transfer");
        }
      };

    } // namespace RecoilPressure
  }   // namespace Simulation
} // namespace MeltPoolDG
