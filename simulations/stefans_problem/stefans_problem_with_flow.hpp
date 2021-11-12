#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

/**
 * This example is derived from
 *
 * Hardt, S., and F. Wondra. "Evaporation model for interfacial flows based on a continuum-field
 * representation of the source terms." Journal of Computational Physics 227.11 (2008): 5871-5895.
 *
 * and represents a simplified example of the denoted "Stefan's Problem 2" with neglection of
 * the temperature field.
 */

namespace MeltPoolDG::Simulation::StefansProblemWithFlow
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;


  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const double x_min,
                    const double x_max,
                    const double y_interface,
                    const double eps)
      : Function<dim>()
      , x_min(x_min)
      , x_max(x_max)
      , y_interface(y_interface)
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const
    {
      Point<dim> left  = dim == 1 ? Point<dim>(y_interface) :
                         dim == 2 ? Point<dim>(x_min, y_interface) :
                                    Point<dim>(x_min, x_min, y_interface);
      Point<dim> right = dim == 1 ? Point<dim>(0) /* not relevant */ :
                         dim == 2 ? Point<dim>(x_max, y_interface) :
                                    Point<dim>(x_max, x_max, y_interface);

      if (p[dim - 1] >= y_interface)
        return -UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
          DistanceFunctions::infinite_line<dim>(p, left, right), eps);
      else
        return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
          DistanceFunctions::infinite_line<dim>(p, left, right), eps);
    }
    double x_min, x_max, y_interface, eps;
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationStefansProblemWithFlow : public SimulationBase<dim>
  {
  public:
    SimulationStefansProblemWithFlow(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.do_simplex || dim == 1)
        {
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      // create mesh
      const Point<dim> bottom_left = dim == 1   ? Point<dim>(y_min) :
                                     (dim == 2) ? Point<dim>(x_min, y_min) :
                                                  Point<dim>(x_min, x_min, y_min);
      const Point<dim> top_right   = dim == 1   ? Point<dim>(y_max) :
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
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }

      // get vertices along the vertical axis on rank 0
      if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        {
          for (unsigned int i = 0; i <= 100.; ++i)
            {
              auto p     = Point<dim>();
              p[dim - 1] = y_min + (y_max - y_min) / 100. * i;
              vertices_along_vertical_axis.emplace_back(p);
            }
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

      if (this->parameters.evapor.ls_value_liquid == -1)
        {
          this->attach_no_slip_boundary_condition(upper_bc, "navier_stokes_u");
          this->attach_open_boundary_condition(lower_bc, "navier_stokes_u");
        }
      else if (this->parameters.evapor.ls_value_liquid == 1)
        {
          this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
          this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");
        }
      else
        AssertThrow(false, ExcNotImplemented());

      if (dim >= 2)
        {
          this->attach_symmetry_boundary_condition(left_bc, "navier_stokes_u");
          this->attach_symmetry_boundary_condition(right_bc, "navier_stokes_u");
        }

      /*
       *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
       *  due to the hyperbolic nature of the analyzed problem)
       *
                    fix/open
       (0,1)  +---------------+ (1,1)
              |    ls=-1      |
              |               |
       sym    |               |  sym
              |               |
              |               |
              |    ls=1       |
              +---------------+
       * (0,1)      fix/open   (1,0)
       */
      if constexpr (dim == 1)
        {
          for (auto &cell : this->triangulation->cell_iterators())
            for (auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[0] == y_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[0] == y_max)
                    face->set_boundary_id(upper_bc);
                }
        }
      else if constexpr (dim == 2)
        {
          for (const auto &cell : this->triangulation->cell_iterators())
            for (const auto &face : cell->face_iterators())
              if ((face->at_boundary()))
                {
                  if (face->center()[1] == y_min)
                    face->set_boundary_id(lower_bc);
                  else if (face->center()[1] == y_max)
                    face->set_boundary_id(upper_bc);
                  else if (face->center()[0] == x_min)
                    face->set_boundary_id(left_bc);
                  else if (face->center()[0] == x_max)
                    face->set_boundary_id(right_bc);
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
      const double eps =
        UtilityFunctions::compute_initial_epsilon<dim>(this->parameters, *this->triangulation);

      AssertThrow(eps > 0, ExcNotImplemented());
      this->attach_initial_condition(
        std::make_shared<InitialValuesLS<dim>>(x_min, x_max, y_interface, eps), "level_set");
      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) final
    {
      dealii::ConditionalOStream pcout(std::cout,
                                       Utilities::MPI::this_mpi_process(this->mpi_communicator) ==
                                         0);
      if (((this->parameters.paraview.do_output) &&
           !(n_time_step % this->parameters.paraview.write_frequency)) ||
          generic_data_out.get_time() == this->parameters.time_stepping.end_time)
        {
          generic_data_out.get_vector("velocity").update_ghost_values();
          generic_data_out.get_vector("heaviside").update_ghost_values();
          generic_data_out.get_vector("pressure").update_ghost_values();

          /*
           * evaluate pressure profile
           */
          if (!remote_point_is_initialized)
            {
              remote_point_evaluation.reinit(vertices_along_vertical_axis,
                                             *this->triangulation,
                                             generic_data_out.get_mapping());
              if (this->parameters.amr.do_amr == false)
                remote_point_is_initialized = true;
            }

          const auto pressure_evaluation_values =
            dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                 generic_data_out.get_dof_handler("pressure"),
                                                 generic_data_out.get_vector("pressure"));

          const auto ls_evaluation_values =
            dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                 generic_data_out.get_dof_handler("heaviside"),
                                                 generic_data_out.get_vector("heaviside"));

          const auto vel_evaluation_values =
            dealii::VectorTools::point_values<dim>(remote_point_evaluation,
                                                   generic_data_out.get_dof_handler("velocity"),
                                                   generic_data_out.get_vector("velocity"));

          const auto analytical_velocity = [&](const double &ls) -> double {
            return this->parameters.evapor.evaporative_mass_flux * (1. - ls) *
                   (1. / this->parameters.material.first.density -
                    1. / this->parameters.material.second.density);
          };

          const auto analytical_pressure = [&](const double &ls) -> double {
            return std::pow(this->parameters.evapor.evaporative_mass_flux, 2) * ls *
                   (1. / this->parameters.material.first.density -
                    1. / this->parameters.material.second.density);
          };

          // write values to file
          if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              const auto file_name = this->parameters.paraview.directory + "/" +
                                     this->parameters.paraview.filename + "_pressure_profile_" +
                                     std::to_string(generic_data_out.get_time()) + ".txt";
              file_pressure_profile.open(file_name);
              file_pressure_profile
                << " coordinate | velocity | analytical velocity | pressure value | analytical pressure value "
                << std::endl;

              for (unsigned int i = 0; i < pressure_evaluation_values.size(); ++i)
                {
                  file_pressure_profile << vertices_along_vertical_axis[i][dim - 1] << " ";

                  if constexpr (dim > 1)
                    file_pressure_profile << vel_evaluation_values[i][dim - 1] << " ";
                  else
                    file_pressure_profile << vel_evaluation_values[i] << " ";
                  file_pressure_profile << analytical_velocity(ls_evaluation_values[i]) << " "
                                        << pressure_evaluation_values[i] << " "
                                        << analytical_pressure(ls_evaluation_values[i])
                                        << std::endl;
                }
              file_pressure_profile.close();
              if constexpr (dim > 1)
                {
                  pcout << "POSTPROCESSOR: min velocity: " << vel_evaluation_values[0][dim - 1]
                        << " (analytical: " << analytical_velocity(ls_evaluation_values[0]) << ")"
                        << std::endl;
                  pcout << "POSTPROCESSOR: max velocity: "
                        << vel_evaluation_values[vel_evaluation_values.size() - 1][dim - 1]
                        << " (analytical: "
                        << analytical_velocity(
                             ls_evaluation_values[vel_evaluation_values.size() - 1])
                        << ")" << std::endl;
                }
              else
                {
                  pcout << "POSTPROCESSOR: min velocity: " << vel_evaluation_values[0]
                        << " (analytical: " << analytical_velocity(ls_evaluation_values[0]) << ")"
                        << std::endl;
                  pcout << "POSTPROCESSOR: max velocity: "
                        << vel_evaluation_values[vel_evaluation_values.size() - 1]
                        << " (analytical: "
                        << analytical_velocity(
                             ls_evaluation_values[vel_evaluation_values.size() - 1])
                        << ")" << std::endl;
                }
              pcout << "POSTPROCESSOR: max pressure: " << pressure_evaluation_values[0]
                    << " (analytical: " << analytical_pressure(ls_evaluation_values[0]) << ")"
                    << std::endl;
              pcout << "POSTPROCESSOR: min pressure: "
                    << pressure_evaluation_values[pressure_evaluation_values.size() - 1]
                    << " (analytical: "
                    << analytical_pressure(
                         ls_evaluation_values[pressure_evaluation_values.size() - 1])
                    << ")" << std::endl;
            }

          generic_data_out.get_vector("level_set").zero_out_ghost_values();
          generic_data_out.get_vector("velocity").zero_out_ghost_values();
          generic_data_out.get_vector("pressure").zero_out_ghost_values();
        }
      n_time_step += 1;
    }


  private:
    const double x_min       = 0.0;
    const double x_max       = 1.0;
    const double y_min       = 0.0;
    const double y_max       = 1.0;
    const double y_interface = 0.5;

    // Postprocessor
    std::ofstream                                   file_velocity_profile;
    std::ofstream                                   file_pressure_profile;
    int                                             n_time_step = 0.0;
    std::vector<Point<dim>>                         vertices_along_vertical_axis;
    bool                                            remote_point_is_initialized = false;
    Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation;
  };
} // namespace MeltPoolDG::Simulation::StefansProblemWithFlow
