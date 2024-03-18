#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>

#include <deal.II/grid/grid_generator.h>

// MeltPoolDG
#include <meltpooldg/flow/characteristic_numbers.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/material/material.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

// c++
#include <cmath>
#include <iostream>

/**
 * This example is derived from
 *
 * Meier et al. 2020
 * "A novel smoothed particle hydrodynamics formulation
 * for thermo-capillary phase change problems with focus on metal additive
 * manufacturing melt pool modeling"
 *
 * and
 *
 * C. Ma, D. Bothe 2011, "Direct numerical simulation of thermocapillary flow
 * based on the Volume of Fluid method"
 *
 *                 T2 = fixed
 *                  no slip
 *            +-------------------+
 *            |                   |
 *            |                   |
 *            |       -----       |
 *   sym      |      --   --      |   sym
 * T Neumann  |     --     --     |   T Neumann
 *            |      --   --      |
 *            |       -----       |
 *            |                   |
 *            +-------------------+
 *                   no slip
 *                  T1 = fixed
 *
 * droplet:
 *    rho    = 250 kg/m³
 *    mu     = 0.012 kg/m/s
 *    lambda = 1.2e-6 W/m/K
 *    cp     = 5e-5 J/kg/K
 *
 * ambient fluid:
 *    rho    = 500 kg/m³
 *    mu     = 0.024 N/m²s
 *    lambda = 2.4e-6 W/m/K
 *    cp     = 1e-4 J/kg/K
 *
 * droplet radius: a=1.44e-3 m
 * surface tension coefficient: 0.01 N/m
 * temperature-dependent surface tension coefficient: 0.002 N/m/K
 * T1 = 290 K
 * T2 = 290 + 4 * a * 200 K/m = 291.152 K
 */

namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;


  static constexpr double a      = 1.44e-3;
  static constexpr double grad_T = 200;

  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const double eps)
      : Function<dim>()
      , distance_sphere(dim == 2 ? Point<dim>(0, 0) : Point<dim>(0, 0, 0), a)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        -distance_sphere.value(p), eps);
    }

  private:
    const Functions::SignedDistance::Sphere<dim> distance_sphere;
    const double                                 eps;
  };


  template <int dim>
  class InitialValuesTemperature : public Function<dim>
  {
  public:
    InitialValuesTemperature()
      : Function<dim>()
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return 290 + grad_T * (p[dim - 1] + 2 * a);
    }
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationThermoCapillaryDroplet : public SimulationBase<dim>
  {
  public:
    SimulationThermoCapillaryDroplet(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        file.open(this->parameters.output.directory + "/" +
                  this->parameters.output.paraview.filename +
                  "_droplet_velocity_over_time_normalized.csv");
      velocity_reference =
        this->parameters.flow.surface_tension.temperature_dependent_surface_tension_coefficient *
        grad_T * a / this->parameters.material.gas.dynamic_viscosity;
      time_reference = a / velocity_reference;

      const auto characteristic_numbers =
        Flow::CharacteristicNumbers<double>(this->parameters.material.gas);
      reynolds_number  = characteristic_numbers.Reynolds(velocity_reference, a);
      mach_number      = characteristic_numbers.Mach(velocity_reference, a);
      capillary_number = characteristic_numbers.capillary(
        velocity_reference, this->parameters.flow.surface_tension.surface_tension_coefficient);
    }

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
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

      if constexpr ((dim == 2) || (dim == 3))
        {
          // create mesh
          const Point<dim> bottom_left =
            (dim == 2) ? Point<dim>(x_min, x_min) : Point<dim>(x_min, x_min, x_min);
          const Point<dim> top_right =
            (dim == 2) ? Point<dim>(x_max, x_max) : Point<dim>(x_max, x_max, x_max);

          if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
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
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      const types::boundary_id lower_bc = 1;
      const types::boundary_id upper_bc = 2;
      const types::boundary_id left_bc  = 3;
      const types::boundary_id right_bc = 4;

      this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
      this->attach_no_slip_boundary_condition(upper_bc, "navier_stokes_u");
      this->attach_symmetry_boundary_condition(left_bc, "navier_stokes_u");
      this->attach_symmetry_boundary_condition(right_bc, "navier_stokes_u");

      this->attach_dirichlet_boundary_condition(lower_bc,
                                                std::make_shared<InitialValuesTemperature<dim>>(),
                                                "heat_transfer");
      this->attach_dirichlet_boundary_condition(upper_bc,
                                                std::make_shared<InitialValuesTemperature<dim>>(),
                                                "heat_transfer");

      if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[1] == x_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[1] == x_max)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == x_min)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
                }
        }
      else if constexpr (dim == 3)
        {
          const types::boundary_id front_bc = 5;
          const types::boundary_id back_bc  = 6;
          this->attach_symmetry_boundary_condition(front_bc, "navier_stokes_u");
          this->attach_symmetry_boundary_condition(back_bc, "navier_stokes_u");
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[1] == x_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[1] == x_max)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == x_min)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
                  else if (face->center()[2] == x_min)
                    face->set_boundary_id(back_bc);
                  else if (face->center()[2] == x_max)
                    face->set_boundary_id(front_bc);
                }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    void
    set_field_conditions() final
    {
      const double eps = this->parameters.ls.reinit.compute_interface_thickness_parameter_epsilon(
        GridTools::minimal_cell_diameter(*this->triangulation) /
        this->parameters.ls.get_n_subdivisions() / std::sqrt(dim));

      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps), "level_set");
      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(),
                                     "heat_transfer");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const final
    {
      if (this->parameters.output.do_user_defined_postprocessing)
        {
          if constexpr (dim > 1)
            {
              dealii::ConditionalOStream pcout(
                std::cout, Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);

              Point<dim> center_of_mass;

              double average_velocity = 0.0;
              double area_of_phase    = 0.0;

              FEValues<dim> level_set_eval(
                generic_data_out.get_mapping(),
                generic_data_out.get_dof_handler("level_set").get_fe(),
                QGauss<dim>(generic_data_out.get_dof_handler("level_set").get_fe().tensor_degree() +
                            1),
                update_quadrature_points | update_JxW_values | update_values);

              const FEValuesExtractors::Vector velocities(0);
              FEValues<dim>                    vel_eval(
                generic_data_out.get_mapping(),
                generic_data_out.get_dof_handler("velocity").get_fe(),
                QGauss<dim>(generic_data_out.get_dof_handler("level_set").get_fe().tensor_degree() +
                            1),
                update_values);

              std::vector<double>         ls_at_q(level_set_eval.n_quadrature_points);
              std::vector<Tensor<1, dim>> vel_at_q(level_set_eval.n_quadrature_points,
                                                   Tensor<1, dim>());

              typename DoFHandler<dim>::active_cell_iterator vel_cell =
                generic_data_out.get_dof_handler("velocity").begin_active();

              for (const auto &cell :
                   generic_data_out.get_dof_handler("level_set").active_cell_iterators())
                {
                  if (cell->is_locally_owned())
                    {
                      level_set_eval.reinit(cell);
                      level_set_eval.get_function_values(generic_data_out.get_vector("level_set"),
                                                         ls_at_q);

                      vel_eval.reinit(vel_cell);
                      vel_eval[velocities].get_function_values(
                        generic_data_out.get_vector("velocity"), vel_at_q);

                      for (const auto q : level_set_eval.quadrature_point_indices())
                        {
                          if (ls_at_q[q] > 0.0)
                            {
                              area_of_phase += level_set_eval.JxW(q);
                              average_velocity += vel_at_q[q][dim - 1] * level_set_eval.JxW(q);
                              for (unsigned int d = 0; d < dim; ++d)
                                center_of_mass[d] +=
                                  level_set_eval.quadrature_point(q)[d] * level_set_eval.JxW(q);
                            }
                        }
                    }
                  ++vel_cell;
                }

              /*
               * area of the phase
               */
              double global_area = Utilities::MPI::sum(area_of_phase, this->mpi_communicator);

              /*
               * centroid position
               */
              Point<dim> global_center_of_mass;
              for (unsigned int d = 0; d < dim; ++d)
                global_center_of_mass[d] =
                  Utilities::MPI::sum(center_of_mass[d], this->mpi_communicator);

              global_center_of_mass /= global_area;

              /*
               * average velocity
               */
              double global_average_velocity =
                Utilities::MPI::sum(average_velocity, this->mpi_communicator);
              global_average_velocity /= global_area;

              /*
               * velocity measured at centroid
               */
              Utilities::MPI::RemotePointEvaluation<dim, dim> cache;
              std::vector<Tensor<1, dim>>                     velocity_of_center =
                VectorTools::point_values<dim>(generic_data_out.get_mapping(),
                                               generic_data_out.get_dof_handler("velocity"),
                                               generic_data_out.get_vector("velocity"),
                                               {global_center_of_mass},
                                               cache);

              pcout << "---------------------------------------------" << std::endl;
              pcout << "    user defined postprocessing" << std::endl;
              pcout << "---------------------------------------------" << std::endl;
              if (print_once)
                {
                  pcout << "reference velocity: " << velocity_reference << std::endl;
                  pcout << "reference time: " << time_reference << std::endl;
                  pcout << "Reynolds number: " << reynolds_number << std::endl;
                  pcout << "Mach number: " << mach_number << std::endl;
                  pcout << "Capillary number: " << capillary_number << std::endl;
                }

              const auto max_vel = generic_data_out.get_vector("velocity").linfty_norm();
              if (file.is_open())
                {
                  pcout << "centroid: " << global_center_of_mass << std::endl;
                  pcout << "vel: " << velocity_of_center[0][dim - 1] << std::endl;
                  if (print_once)
                    file << "time,y_center,t/tr,u/ur,u_max/ur,u_avg/ur" << std::endl;

                  file << generic_data_out.get_time() << ", " << global_center_of_mass[dim - 1]
                       << ", " << generic_data_out.get_time() / time_reference << ", "
                       << velocity_of_center[0][dim - 1] / velocity_reference << ", "
                       << max_vel / velocity_reference << ", "
                       << global_average_velocity / velocity_reference << std::endl;
                }
              pcout << "---------------------------------------------" << std::endl;
              pcout << "---------------------------------------------" << std::endl;
            }
          print_once = false;
        }
    }

  private:
    const double x_min = -2 * a;
    const double x_max = 2 * a;
    double       velocity_reference;
    double       time_reference;
    double       reynolds_number;
    double       mach_number;
    double       capillary_number;
    // postprocessing
    mutable std::ofstream file;
    mutable bool          print_once = true;
  };
} // namespace MeltPoolDG::Simulation::ThermoCapillaryDroplet
