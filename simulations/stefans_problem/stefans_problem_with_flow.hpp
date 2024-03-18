#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/evaporation/evaporation_model_constant.hpp>
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>

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
      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP || dim == 1)
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

      // create mesh
      const Point<dim> bottom_left = dim == 1   ? Point<dim>(y_min) :
                                     (dim == 2) ? Point<dim>(x_min, y_min) :
                                                  Point<dim>(x_min, x_min, y_min);
      const Point<dim> top_right   = dim == 1   ? Point<dim>(y_max) :
                                     (dim == 2) ? Point<dim>(x_max, y_max) :
                                                  Point<dim>(x_max, x_max, y_max);

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          std::vector<unsigned int> subdivisions(
            dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;

          GridGenerator::subdivided_hyper_rectangle_with_simplices(
            *this->triangulation, subdivisions, bottom_left, top_right, true /*colorize*/);
        }
      else
        {
          GridGenerator::hyper_rectangle(*this->triangulation,
                                         bottom_left,
                                         top_right,
                                         true /*colorize*/);
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
      // faces in dim-1 direction
      const types::boundary_id lower_bc = 2 * (dim - 1);
      const types::boundary_id upper_bc = 2 * (dim - 1) + 1;

      this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
      this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");

      // collect boundary ids of side walls
      std::vector<types::boundary_id> side_walls;

      for (unsigned int i = 0; i < 2 * (dim - 1); ++i)
        side_walls.push_back(i);

      for (const auto &s : side_walls)
        this->attach_symmetry_boundary_condition(s, "navier_stokes_u");
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<Functions::SignedDistance::Plane<dim>>(
                                       Point<dim>::unit_vector(dim - 1) * y_interface,
                                       -Point<dim>::unit_vector(dim - 1)),
                                     "signed_distance");
      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const final
    {
      dealii::ConditionalOStream pcout(std::cout,
                                       Utilities::MPI::this_mpi_process(this->mpi_communicator) ==
                                         0);
      if (!(n_time_step % this->parameters.output.write_frequency) ||
          generic_data_out.get_time() == this->parameters.time_stepping.end_time)
        {
          std::cout.precision(3);
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

          const auto m_dot =
            Evaporation::EvaporationModelConstant(this->parameters.evapor.analytical.function);

          const auto analytical_velocity = [&](const double &ls) -> double {
            return m_dot.local_compute_evaporative_mass_flux(generic_data_out.get_time()) *
                   (1. - ls) *
                   (1. / this->parameters.material.gas.density -
                    1. / this->parameters.material.liquid.density);
          };

          const auto analytical_pressure = [&](const double &ls) -> double {
            return std::pow(m_dot.local_compute_evaporative_mass_flux(generic_data_out.get_time()),
                            2) *
                   ls *
                   (1. / this->parameters.material.gas.density -
                    1. / this->parameters.material.liquid.density);
          };

          // write values to file
          if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              if (this->parameters.output.do_user_defined_postprocessing)
                {
                  const auto file_name = this->parameters.output.directory + "/" +
                                         this->parameters.output.paraview.filename +
                                         "_pressure_profile_" +
                                         std::to_string(generic_data_out.get_time()) + ".txt";
                  file_pressure_profile.open(file_name);
                  file_pressure_profile
                    << "% coordinate | velocity | analytical velocity | pressure value | analytical pressure value "
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
                }

              // console output
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
    mutable std::ofstream                                   file_velocity_profile;
    mutable std::ofstream                                   file_pressure_profile;
    mutable int                                             n_time_step = 0.0;
    mutable std::vector<Point<dim>>                         vertices_along_vertical_axis;
    mutable bool                                            remote_point_is_initialized = false;
    mutable Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation;
  };
} // namespace MeltPoolDG::Simulation::StefansProblemWithFlow
