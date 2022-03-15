#pragma once

// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace RecoilPressure
    {
      using namespace dealii;

      template <int dim>
      class InitialConditionTemperature : public Function<dim>
      {
      public:
        InitialConditionTemperature(const double T_initial_bottom,
                                    const double T_initial_top,
                                    const double y_min,
                                    const double y_max)
          : Function<dim>()
          , T_initial_bottom(T_initial_bottom)
          , T_initial_top(T_initial_top)
          , y_min(y_min)
          , grad_T((T_initial_top - T_initial_bottom) / (y_max - y_min))
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const
        {
          if (T_initial_top == T_initial_bottom)
            return T_initial_top;
          else

            return T_initial_bottom + grad_T * (p[dim - 1] - y_min);
        }

        const double T_initial_bottom;
        const double T_initial_top;
        const double y_min;
        const double grad_T;
      };

      template <int dim>
      class SimulationRecoilPressure : public SimulationBase<dim>
      {
      private:
        double       domain_x_min                   = 0;
        double       domain_x_max                   = 0;
        double       domain_y_min                   = 0;
        double       domain_y_max                   = 0;
        bool         periodic_boundary              = false;
        bool         evaporation_boundary           = false;
        unsigned int n_local_refinement             = 0;
        double       T_initial_top                  = 500;
        double       T_initial_bottom               = T_initial_top;
        std::string  local_refinement_1_bottom_left = "";
        std::string  local_refinement_1_top_right   = "";
        std::string  local_refinement_2_bottom_left = "";
        std::string  local_refinement_2_top_right   = "";

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
            prm.add_parameter("n local refinement",
                              n_local_refinement,
                              "number of (additional to the global) refinements for local region.");
            prm.add_parameter("local refinement 1 bottom left",
                              local_refinement_1_bottom_left,
                              "Bottom left point of locally refined region.");
            prm.add_parameter("local refinement 1 top right",
                              local_refinement_1_top_right,
                              "Bottom left point of locally refined region.");
            prm.add_parameter("local refinement 2 bottom left",
                              local_refinement_2_bottom_left,
                              "Bottom left point of locally refined region.");
            prm.add_parameter("local refinement 2 top right",
                              local_refinement_2_top_right,
                              "Bottom left point of locally refined region.");
            prm.add_parameter(
              "periodic boundary",
              periodic_boundary,
              "Set this parameter to true if the domain should be periodic in x direction.");
            prm.add_parameter(
              "evaporation boundary",
              evaporation_boundary,
              "Set this Parameter to true if the upper boundary of the domain should be open "
              "to enable an outward mass flow.");
            prm.enter_subsection("initial temperature");
            {
              prm.add_parameter("top",
                                T_initial_top,
                                "Set the initial temperature on the top boundary.");
              prm.add_parameter("bottom",
                                T_initial_bottom,
                                "Set the initial temperature on the bottom boundary.");
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }

        void
        create_spatial_discretization() override
        {
          if (this->parameters.base.do_simplex || dim == 1)
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
          // create mesh
          const Point<dim> bottom_left = (dim == 1) ? Point<dim>(y_min) :
                                         (dim == 2) ? Point<dim>(x_min, y_min) :
                                                      Point<dim>(x_min, x_min, y_min);
          const Point<dim> top_right   = (dim == 1) ? Point<dim>(y_max) :
                                         (dim == 2) ? Point<dim>(x_max, y_max) :
                                                      Point<dim>(x_max, x_max, y_max);

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
            }
        }

        void
        set_boundary_conditions() override
        {
          /*
           *                  upper_bc
           *            +-------------------+
           *            |                   |
           *            |      ls = -1      |
           *            |                   |
           *   left_bc  |-------------------|  right_bc
           *            |                   |
           *            |      ls = 1       |
           *            |                   |
           *            +-------------------+
           *                   lower_bc
           *
           * -----------------------------------------------------------------------------
           * evaporation boundary    = false
           * periodic boundary = false
           *                     |   temperature  | velocity |    pressure    | level set
           * left_bc, right_bc   |  adiabatic     | no-slip  |       -        |     -
           * lower_bc            |  T = T_initial | no-slip  | fixed constant |     -
           * upper_bc            |  T = T_initial | no-slip  |       -        |     -
           * -----------------------------------------------------------------------------
           * evaporation boundary    = false
           * periodic boundary = true
           *                     |   temperature  | velocity |    pressure    | level set
           * left_bc, right_bc   |    periodic    | periodic |    periodic    | periodic
           * lower_bc            |  T = T_initial | no-slip  | fixed constant |     -
           * upper_bc            |  T = T_initial | no-slip  |       -        |     -
           * -----------------------------------------------------------------------------
           * evaporation boundary    = true
           * periodic boundary = false
           *                     |   temperature  | velocity |    pressure    | level set
           * left_bc, right_bc   |  adiabatic     | symmetry |       -        |     -
           * lower_bc            |  T = T_initial |   open   |       -        |     -
           * upper_bc            |  T = T_initial | no-slip  |       -        |     -
           * -----------------------------------------------------------------------------
           * evaporation boundary    = true
           * periodic boundary = true
           *                     |   temperature  | velocity |    pressure    | level set
           * left_bc, right_bc   |    periodic    | periodic |    periodic    | periodic
           * lower_bc            |  T = T_initial |   open   |       -        |     -
           * upper_bc            |  T = T_initial | no-slip  |       -        |     -
           * -----------------------------------------------------------------------------
           *
           * note: In the 3D recoil pressure simulation, the front and back boundaries are treated
           * the same as the left and right boundaries.
           */
          const types::boundary_id lower_bc = 1;
          const types::boundary_id upper_bc = 2;
          const types::boundary_id left_bc  = 3;
          const types::boundary_id right_bc = 4;
          const types::boundary_id front_bc = 5;
          const types::boundary_id back_bc  = 6;

          if (dim == 1)
            {
              for (const auto &cell : this->triangulation->cell_iterators())
                for (const auto &face : cell->face_iterators())
                  if ((face->at_boundary()))
                    {
                      if (face->center()[0] == domain_y_min)
                        face->set_boundary_id(lower_bc);
                      else if (face->center()[0] == domain_y_max)
                        face->set_boundary_id(upper_bc);
                    }
            }
          else if ((dim == 2) || (dim == 3))
            {
              for (const auto &cell : this->triangulation->cell_iterators())
                for (const auto &face : cell->face_iterators())
                  if ((face->at_boundary()))
                    {
                      if (face->center()[0] == domain_x_min)
                        face->set_boundary_id(left_bc);
                      else if (face->center()[0] == domain_x_max)
                        face->set_boundary_id(right_bc);
                      else if (dim == 2)
                        {
                          if (face->center()[1] == domain_y_min)
                            face->set_boundary_id(lower_bc);
                          else if (face->center()[1] == domain_y_max)
                            face->set_boundary_id(upper_bc);
                        }
                      else // dim == 3
                        {
                          if (face->center()[1] == domain_x_min)
                            face->set_boundary_id(back_bc);
                          else if (face->center()[1] == domain_x_max)
                            face->set_boundary_id(front_bc);
                          else if (face->center()[2] == domain_y_min)
                            face->set_boundary_id(lower_bc);
                          else if (face->center()[2] == domain_y_max)
                            face->set_boundary_id(upper_bc);
                        }
                    }
            }
          else
            AssertThrow(false, ExcNotImplemented());

          if (periodic_boundary)
            {
              if (dim >= 2)
                this->attach_periodic_boundary_condition(left_bc, right_bc, 0);
              if (dim == 3)
                this->attach_periodic_boundary_condition(front_bc, back_bc, 1);
            }

          /*
           * BC for two-phase flow
           */
          if (this->parameters.base.problem_name == ProblemType::melt_pool)
            {
              this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
              if (evaporation_boundary)
                {
                  this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");
                  if (!periodic_boundary)
                    {
                      this->attach_symmetry_boundary_condition(left_bc, "navier_stokes_u");
                      this->attach_symmetry_boundary_condition(right_bc, "navier_stokes_u");
                      if (dim == 3)
                        {
                          this->attach_symmetry_boundary_condition(front_bc, "navier_stokes_u");
                          this->attach_symmetry_boundary_condition(back_bc, "navier_stokes_u");
                        }
                    }
                }
              else // no evaporation
                {
                  // The fix pressure constant condition can be set on any boundary (that is not a
                  // periodic boundary).
                  this->attach_fix_pressure_constant_condition(lower_bc, "navier_stokes_p");
                  this->attach_no_slip_boundary_condition(upper_bc, "navier_stokes_u");
                  if (!periodic_boundary)
                    {
                      this->attach_no_slip_boundary_condition(left_bc, "navier_stokes_u");
                      this->attach_no_slip_boundary_condition(right_bc, "navier_stokes_u");
                      if (dim == 3)
                        {
                          this->attach_no_slip_boundary_condition(front_bc, "navier_stokes_u");
                          this->attach_no_slip_boundary_condition(back_bc, "navier_stokes_u");
                        }
                    }
                }
            }
          else
            AssertThrow(false, ExcNotImplemented());

          /*
           * BC for heat transfer
           */
          if (this->parameters.laser.heat_source_model != LaserHeatSourceModel::Analytical)
            {
              if (evaporation_boundary && this->parameters.heat.convection_coefficient > 0)
                {
                  this->attach_convection_boundary_condition(upper_bc, "heat_transfer");
                }
              else
                this->attach_dirichlet_boundary_condition(
                  upper_bc,
                  std::make_shared<Functions::ConstantFunction<dim>>(T_initial_top),
                  "heat_transfer");

              this->attach_dirichlet_boundary_condition(
                lower_bc,
                std::make_shared<Functions::ConstantFunction<dim>>(T_initial_bottom),
                "heat_transfer");
            }

          if (!this->parameters.base.do_simplex)
            this->triangulation->refine_global(this->parameters.base.global_refinements);

          /*
           * locally refined region described by max. 2 bounding boxes
           *
           *
           *  +-------------------------+
           *  |                         |
           *  |         +--------+      |
           *  |         |refine 1|      |
           *  |         +--------+      |
           *  |                         |
           *  |                         |
           *  |         +--------+      |
           *  |         |refine 2|      |
           *  |         +--------+      |
           *  |                         |
           *  |                         |
           *  +-------------------------+
           *
           */
          if (n_local_refinement > 0)
            {
              if constexpr (dim == 2)
                {
                  // 1. region
                  AssertThrow(local_refinement_1_bottom_left.size() > 0 &&
                                local_refinement_1_bottom_left.size() > 0,
                              ExcMessage("The points of the refinement region must be specified."));
                  const auto bl = MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
                    local_refinement_1_bottom_left);
                  const auto tr = MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
                    local_refinement_1_top_right);
                  const auto refinement_region = BoundingBox<dim>({bl, tr});

                  // 2. region (optional)
                  Point<dim> bl_2;
                  Point<dim> tr_2;
                  if (local_refinement_2_bottom_left.size() > 0 &&
                      local_refinement_2_top_right.size() > 0)
                    {
                      bl_2 = MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
                        local_refinement_2_bottom_left);
                      tr_2 = MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
                        local_refinement_2_top_right);
                    }
                  const auto refinement_region_2 = BoundingBox<dim>({bl_2, tr_2});

                  for (unsigned int j = 0; j < n_local_refinement; ++j)
                    {
                      for (auto &cell : this->triangulation->active_cell_iterators())
                        {
                          if (cell->is_locally_owned())
                            {
                              for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                                if (refinement_region.point_inside(cell->vertex(i)) ||
                                    refinement_region_2.point_inside(cell->vertex(i)))
                                  {
                                    cell->set_refine_flag();
                                    break;
                                  }
                            }
                        }
                      this->triangulation->execute_coarsening_and_refinement();
                    }
                }
              else
                AssertThrow(false, ExcNotImplemented());
            }
        }

        void
        set_field_conditions() override
        {
          auto laser_center = MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
            this->parameters.laser.center);

          this->attach_initial_condition(std::make_shared<Functions::SignedDistance::Plane<dim>>(
                                           Point<dim>::unit_vector(dim - 1) * laser_center[dim - 1],
                                           -Point<dim>::unit_vector(dim - 1)),
                                         "signed_distance");
          this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                           new Functions::ZeroFunction<dim>(dim)),
                                         "navier_stokes_u");
          if (this->parameters.laser.heat_source_model != LaserHeatSourceModel::Analytical)
            this->attach_initial_condition(
              std::make_shared<InitialConditionTemperature<dim>>(
                T_initial_bottom, T_initial_top, domain_y_min, domain_y_max),
              "heat_transfer");
        }
      };

    } // namespace RecoilPressure
  }   // namespace Simulation
} // namespace MeltPoolDG
