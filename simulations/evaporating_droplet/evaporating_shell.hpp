#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <meltpooldg/interface/simulation_base.hpp>

#include <iostream>

/**
 * This example provides a simplified version of the EvaporatingDroplet simulation,
 * where only the vapor domain is computed. To this end, a circular shell is modelled
 * with a constant radial velocity at the interior edge and open boundary conditions
 * at the exterior edge. This leads to axisymmetric conditions. In this example, the
 * inflow should mimic the evaporative flow that escapes from the droplet into the
 * vapor domain. By comparison with the analytical solution, the predicted pressure
 * and velocity field can be verified.
 */

namespace MeltPoolDG::Simulation::EvaporatingShell
{
  using namespace dealii;

  // velocity prescribed at the interior edge
  static double velocity = 0.0;
  // inner radius of the circular shell
  static double inner_radius = 0.5;
  // outer radius of the circular shell
  static double outer_radius = 2.0;

  template <int dim>
  class RadialBoundaryVelocity : public Function<dim>
  {
  public:
    RadialBoundaryVelocity(const Point<dim> &center)
      : Function<dim>(dim)
      , center(center)
    {}

    double
    value(const Point<dim> &p, const unsigned int comp) const
    {
      const double radius = p.distance(center);
      const double v      = velocity;

      const auto n = p - center;

      if constexpr (dim == 1)
        return v;
      else if constexpr (dim == 2)
        {
          if (comp == 0)
            return v * n[0] / radius;
          else if (comp == 1)
            return v * n[1] / radius;
          else
            AssertThrow(false, ExcNotImplemented());
        }
      else if constexpr (dim == 3)
        {
          const double phi   = std::acos(n[2] / radius);
          const double theta = std::atan2(n[1], n[0]);

          if (comp == 0)
            return v * std::cos(theta) * std::sin(phi);
          else if (comp == 1)
            return v * std::sin(theta) * std::sin(phi);
          else if (comp == 2)
            return v * std::cos(phi);
          else
            AssertThrow(false, ExcNotImplemented());
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }

  private:
    const Point<dim> center;
  };

  template <int dim>
  class SimulationEvaporatingShell : public SimulationBase<dim>
  {
  public:
    SimulationEvaporatingShell(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("velocity",
                          velocity,
                          "Radial velocity on the interior edge of the shell.");
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.do_simplex)
        {
          this->triangulation =
            std::make_shared<parallel::shared::Triangulation<dim>>(this->mpi_communicator);
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      GridGenerator::hyper_shell(
        *this->triangulation, center, inner_radius, outer_radius, 0, true /*colorize*/);

      if (this->parameters.base.do_simplex == false)
        this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      // outer boundary
      this->attach_open_boundary_condition(1, "navier_stokes_u");

      // inner boundary
      const auto dirichlet = std::make_shared<RadialBoundaryVelocity<dim>>(center);
      this->attach_dirichlet_boundary_condition(0, dirichlet, "navier_stokes_u");
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(-1),
                                     "level_set");
      this->attach_initial_condition(std::make_shared<RadialBoundaryVelocity<dim>>(center),
                                     "navier_stokes_u");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const final
    {
      // Compare numerical vs. analytical solution based on the last computed time-step.
      // This is sufficient since the test case is stationary.
      if ((n_time_step == this->parameters.time_stepping.max_n_steps) ||
          generic_data_out.get_time() == this->parameters.time_stepping.end_time)
        {
          generic_data_out.get_vector("velocity").update_ghost_values();
          generic_data_out.get_vector("pressure").update_ghost_values();

          const auto analytical_velocity = [&](const double &r) -> double {
            return velocity * inner_radius / r;
          };

          const auto analytical_pressure = [&](const double &r) -> double {
            const double rho = this->parameters.material.first.density;
            const double mu  = this->parameters.material.first.viscosity;
            const double u0  = velocity;
            return rho * u0 * u0 * inner_radius * inner_radius * 0.5 *
                     (1. / std::pow(outer_radius, 2) - 1 / (r * r)) -
                   2. * mu * u0 * inner_radius / std::pow(outer_radius, 2);
          };

          // generate number of request points and calculate analytical solution
          std::vector<Point<dim>> req_points;
          TableHandler            table;
          if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              const std::vector<double> req_radii = {inner_radius,
                                                     (inner_radius + outer_radius) * 0.5,
                                                     outer_radius};

              for (const auto r : req_radii)
                {
                  auto p = Point<dim>();
                  p[0]   = r;
                  req_points.emplace_back(p);
                  table.add_value("analytical pressure", analytical_pressure(r));
                  table.add_value("analytical radial velocity", analytical_velocity(r));
                }
            }

          // read numerical results at request points
          Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation;
          remote_point_evaluation.reinit(req_points,
                                         *this->triangulation,
                                         generic_data_out.get_mapping());

          const auto pressure_vals =
            dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                 generic_data_out.get_dof_handler("pressure"),
                                                 generic_data_out.get_vector("pressure"));

          const auto vel_vals =
            dealii::VectorTools::point_values<dim>(remote_point_evaluation,
                                                   generic_data_out.get_dof_handler("velocity"),
                                                   generic_data_out.get_vector("velocity"));
          if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              for (unsigned int i = 0; i < vel_vals.size(); ++i)
                {
                  table.add_value("pressure", pressure_vals[i]);
                  if constexpr (dim == 1)
                    table.add_value("radial velocity", vel_vals[i]);
                  else
                    table.add_value("radial velocity", vel_vals[i][0]);
                }

              table.set_scientific("analytical radial velocity", true);
              table.set_scientific("analytical pressure", true);
              table.set_scientific("radial velocity", true);
              table.set_scientific("pressure", true);
              std::cout << std::endl;
              table.write_text(std::cout);
              std::cout << std::endl;
            }

          generic_data_out.get_vector("velocity").zero_out_ghost_values();
          generic_data_out.get_vector("pressure").zero_out_ghost_values();
        }
      n_time_step += 1;
    }

    const Point<dim> center;

    // needed for postprocessing
    mutable unsigned int n_time_step = 0.0;
  };

} // namespace MeltPoolDG::Simulation::EvaporatingShell
