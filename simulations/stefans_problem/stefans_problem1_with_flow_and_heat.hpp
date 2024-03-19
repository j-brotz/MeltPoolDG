#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>

#include <boost/math/tools/roots.hpp>

#include <cmath>
#include <iostream>
#include <string>

/**
 *
 * This example is derived from
 *
 * Hardt, S., and F. Wondra. "Evaporation model for interfacial flows based on a continuum-field
 * representation of the source terms." Journal of Computational Physics 227.11 (2008): 5871-5895.
 *
 * and represents the example denoted as "Stefan's Problem 1".
 *
 * The parameters listed in the paper are:
 *
 * domain size [0.0, 0.001] discretized in 1d by 1000 cells
 *
 * Initial interface location: 1e-6 m
 *
 * boiling temperature:             373.15 K
 * temperature at the hot wall: 383.15 K
 *
 * gas (vapor) phase:
 *    -- density:                1 kg/m^3
 *    -- viscosity:              0.0001 Pa/s
 *    -- thermal_conductivity:   1e-2 W/(mK)
 *    -- specific_heat_capacity: 1000 J/(kgK)
 *
 * liquid phase:
 *    -- density:                1 kg/m^3
 *    -- viscosity:              0.01 Pa/s
 *    -- thermal_conductivity:   1 W/(mK) (Note: thermal diffusivity of the liquid phase was
 * increased by order of magnitudes to obtain a constant temperature in the liquid phase)
 *    -- specific_heat_capacity: 1000 J/(kgK)
 *
 * Enthalpy of evaporation: 10^6 J/kg
 *
 * NOTE: Due to the equal densities in the two phases, no flow velocities will be induced.
 *
 * NOTE: In the publication, they did not use the evaporative mass flux calculated according to
 * Schrage's theory. We used the model by Hardt and Wondra and calibrated the evaporation
 * coefficient.
 */

namespace MeltPoolDG::Simulation::StefansProblem1WithFlowAndHeat
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static constexpr double x_min = 0.0;
  static constexpr double y_min = 0.0;
  static constexpr double y_max = 1.e-3;


  namespace AnalyticalSolution
  {

    template <typename number>
    double
    compute_beta(const Parameters<number> &parameters, const double T_wall)
    {
      const auto beta_func = [&](double beta) -> double {
        return beta * std::exp(beta * beta) * erf(beta) -
               parameters.material.gas.specific_heat_capacity *
                 (T_wall - parameters.material.boiling_temperature) /
                 (parameters.material.latent_heat_of_evaporation * std::sqrt(numbers::PI));
      };

      std::pair<double, double> result =
        boost::math::tools::bisect(beta_func, 1e-6, 100., [&](double min, double max) -> bool {
          return std::abs(max - min) <= 1e-10;
        });

      return (result.first + result.second) / 2;
    }

    template <typename number>
    double
    analytical_interface_location(const Parameters<number> &parameters,
                                  const double              T_wall,
                                  const double              time,
                                  const double              beta = -1.0)
    {
      if (beta == -1.0)
        compute_beta(parameters, T_wall);

      return 2. * beta *
             std::sqrt(
               parameters.material.gas.thermal_conductivity /
               (parameters.material.gas.density * parameters.material.gas.specific_heat_capacity) *
               time);
    }

    template <typename number>
    double
    analytical_temperature(const Parameters<number> &parameters,
                           const double              T_wall,
                           const double              time,
                           const double              x,
                           const double              beta = -1.0)
    {
      if (beta == -1.0)
        compute_beta(parameters, T_wall);

      const double diffusivity =
        parameters.material.gas.thermal_conductivity /
        (parameters.material.gas.density * parameters.material.gas.specific_heat_capacity);

      if (time == 0)
        return parameters.material.boiling_temperature;
      else
        return std::max(T_wall + (parameters.material.boiling_temperature - T_wall) *
                                   erf(x / (2. * std::sqrt(diffusivity * time))) / erf(beta),
                        parameters.material.boiling_temperature);
    }
  } // namespace AnalyticalSolution

  template <int dim>
  class InitialValuesTemperature : public Function<dim>
  {
  public:
    InitialValuesTemperature(const double T_sat, const double T_wall, const double y_interface)
      : Function<dim>()
      , T_sat(T_sat)
      , T_wall(T_wall)
      , y_interface(y_interface)

    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const final
    {
      return std::max(T_wall - (T_wall - T_sat) * p[dim - 1] / y_interface, T_sat);
    }

    const double T_sat, T_wall, y_interface;
  };
  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationStefansProblem1WithFlowAndHeat : public SimulationBase<dim>
  {
  public:
    SimulationStefansProblem1WithFlowAndHeat(std::string    parameter_file,
                                             const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
      , x_max(y_max / std::pow(dim, this->parameters.base.global_refinements))
      , remote_point_evaluation(1e-6, true)
    {
      AssertThrow(y_interface >= y_min && y_interface <= y_max,
                  ExcMessage(
                    "The location of the initial interface must be between y_min and y_max."));

      file_name_level_set_contour = this->parameters.output.directory + "/" +
                                    this->parameters.output.paraview.filename +
                                    "_level_set_contour_over_time.txt";
    }

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("y interface", y_interface, "initial interface location");
        prm.add_parameter("T wall", T_wall, "heated temperature of the wall");
      }
      prm.leave_subsection();
    }

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

      const unsigned int n_elements_per_edge =
        dim > 1 ? std::pow(dim, this->parameters.base.global_refinements) :
                  this->parameters.base.global_refinements;

      std::vector<unsigned int> refinements(dim, 1);
      refinements[dim - 1] = n_elements_per_edge;

      // create mesh
      const Point<dim> bottom_left = dim == 1   ? Point<dim>(y_min) :
                                     (dim == 2) ? Point<dim>(x_min, y_min) :
                                                  Point<dim>(x_min, x_min, y_min);
      const Point<dim> top_right   = dim == 1   ? Point<dim>(y_max) :
                                     (dim == 2) ? Point<dim>(x_max, y_max) :
                                                  Point<dim>(x_max, x_max, y_max);

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        {
          // create mesh
          std::vector<unsigned int> subdivisions(
            dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
          subdivisions[dim - 1] *= 2;

          GridGenerator::subdivided_hyper_rectangle_with_simplices(
            *this->triangulation, subdivisions, bottom_left, top_right, true /*colorize*/);
        }
      else
        {
          GridGenerator::subdivided_hyper_rectangle(
            *this->triangulation, refinements, bottom_left, top_right, true /*colorize*/);
        }


      // get vertices along the vertical axis on rank 0
      if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
        {
          const unsigned int n_elements = std::min<double>(200, n_elements_per_edge);
          for (unsigned int i = 0; i <= n_elements; ++i)
            {
              auto p     = Point<dim>();
              p[dim - 1] = y_min + (y_max - y_min) / n_elements * i;
              vertices_along_vertical_axis.emplace_back(p);
            }
        }
    }

    void
    set_boundary_conditions() final
    {
      // faces in dim-1 direction
      const types::boundary_id bottom_bc = 2 * (dim - 1);
      const types::boundary_id top_bc    = bottom_bc + 1;

      // lower part = gas; upper part = liquid
      this->attach_dirichlet_boundary_condition(
        bottom_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1.0), "level_set");
      this->attach_dirichlet_boundary_condition(
        bottom_bc, std::make_shared<Functions::ConstantFunction<dim>>(T_wall), "heat_transfer");
      this->attach_dirichlet_boundary_condition(top_bc,
                                                std::make_shared<Functions::ConstantFunction<dim>>(
                                                  this->parameters.material.boiling_temperature),
                                                "heat_transfer");

      // dummy BC for Navier-Stokes
      this->attach_no_slip_boundary_condition(bottom_bc, "navier_stokes_u");
      this->attach_fix_pressure_constant_condition(top_bc, "navier_stokes_p");

      // collect boundary ids of side walls
      std::vector<types::boundary_id> side_walls;

      if (dim > 1)
        for (unsigned int i = 0; i < 2 * (dim - 1); ++i)
          side_walls.push_back(i);

      // set PBC on side walls
      if (dim >= 2)
        this->attach_periodic_boundary_condition(side_walls[0], side_walls[1], 0);
      if (dim == 3)
        this->attach_periodic_boundary_condition(side_walls[2], side_walls[3], 1);
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");
      this->attach_initial_condition(std::make_shared<Functions::SignedDistance::Plane<dim>>(
                                       Point<dim>::unit_vector(dim - 1) * y_interface,
                                       Point<dim>::unit_vector(dim - 1)),
                                     "signed_distance");

      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(
                                       this->parameters.material.boiling_temperature,
                                       T_wall,
                                       y_interface),
                                     "heat_transfer");
    }

    void
    do_postprocessing(const GenericDataOut<dim> &generic_data_out) const final
    {
      // first time postprocessing
      if (beta == -1.0)
        {
          beta = AnalyticalSolution::compute_beta(this->parameters, T_wall);
          if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              file_level_set_contour.open(file_name_level_set_contour);
              file_level_set_contour
                << "time interface_location interface_temperature analytical_interface_location(beta=" +
                     std::to_string(beta) + ")"
                << std::endl;
              file_level_set_contour.close();
            }
        }

      if (((this->parameters.output.paraview.enable) &&
           !(n_time_step % this->parameters.output.write_frequency)) ||
          generic_data_out.get_time() == this->parameters.time_stepping.end_time)
        {
          generic_data_out.get_vector("level_set").update_ghost_values();
          generic_data_out.get_vector("temperature").update_ghost_values();

          /*
           * evaluate temperature profile
           */
          if (!remote_point_is_initialized)
            {
              remote_point_evaluation.reinit(vertices_along_vertical_axis,
                                             *this->triangulation,
                                             generic_data_out.get_mapping());
              remote_point_is_initialized = true;
            }

          const auto temperature_evaluation_values =
            dealii::VectorTools::point_values<1>(remote_point_evaluation,
                                                 generic_data_out.get_dof_handler("temperature"),
                                                 generic_data_out.get_vector("temperature"));

          // write values to file
          if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              const auto file_name = this->parameters.output.directory + "/" +
                                     this->parameters.output.paraview.filename +
                                     "_temperature_profile_" +
                                     std::to_string(generic_data_out.get_time()) + ".txt";
              file_temperature_profile.open(file_name);
              file_temperature_profile
                << "coordinate temperature analytical_temperature(beta=" + std::to_string(beta) +
                     ")"
                << std::endl;

              for (unsigned int i = 0; i < temperature_evaluation_values.size(); ++i)
                {
                  file_temperature_profile << vertices_along_vertical_axis[i][dim - 1] << " "
                                           << temperature_evaluation_values[i] << " "
                                           << AnalyticalSolution::analytical_temperature(
                                                this->parameters,
                                                T_wall,
                                                generic_data_out.get_time(),
                                                vertices_along_vertical_axis[i][dim - 1],
                                                beta)
                                           << std::endl;
                }
              file_temperature_profile.close();
            }

          /*
           * evaluate location of and temperature at level set == 0
           */
          FEPointEvaluation<1, dim> temperature_eval(
            generic_data_out.get_mapping(),
            generic_data_out.get_dof_handler("temperature").get_fe(),
            update_values);

          std::vector<std::pair<Point<dim>, double>> vertices_and_temperatures;

          std::vector<double>                  buffer;
          std::vector<types::global_dof_index> local_dof_indices;

          LevelSet::Tools::evaluate_at_interface<dim>(
            generic_data_out.get_dof_handler("level_set"),
            generic_data_out.get_mapping(),
            generic_data_out.get_vector("level_set"),
            [&](const auto                  &cell,
                const auto                  &points_real,
                const auto                  &points,
                [[maybe_unused]] const auto &weights) {
              local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
              buffer.resize(cell->get_fe().n_dofs_per_cell());
              cell->get_dof_indices(local_dof_indices);

              const unsigned int n_points = points.size();

              const ArrayView<const Point<dim>> unit_points(points.data(), n_points);
              temperature_eval.reinit(cell, unit_points);

              cell->get_dof_values(generic_data_out.get_vector("temperature"),
                                   buffer.begin(),
                                   buffer.end());

              // evaluate temperature and level set points
              temperature_eval.evaluate(buffer, EvaluationFlags::values);
              for (unsigned int q = 0; q < n_points; ++q)
                {
                  vertices_and_temperatures.emplace_back(points_real[q],
                                                         temperature_eval.get_value(q));
                }
            },
            0.0, /*contour value*/
            3 /*n_subdivisions*/);

          // collect result on rank 0 to write them to file
          const auto vertices_and_temperatures_all =
            Utilities::MPI::reduce<std::vector<std::pair<Point<dim>, double>>>(
              vertices_and_temperatures, this->mpi_communicator, [](const auto &a, const auto &b) {
                auto result = a;
                result.insert(result.end(), b.begin(), b.end());
                return result;
              });

          // compute analytical solution
          const double interface_analytical = AnalyticalSolution::analytical_interface_location(
            this->parameters, T_wall, generic_data_out.get_time(), beta);
          // write values to file
          if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
            {
              file_level_set_contour.open(file_name_level_set_contour, std::fstream::app);
              file_level_set_contour
                << generic_data_out.get_time() << " "
                << vertices_and_temperatures_all[0].first[dim - 1] - y_interface << " "
                << vertices_and_temperatures_all[0].second << " " << interface_analytical
                << std::endl;
              file_level_set_contour.close();
            }

          dealii::ConditionalOStream pcout(
            std::cout, Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0);
          pcout << "interface_numerical " << vertices_and_temperatures_all[0].first[dim - 1]
                << " interface_analytical " << interface_analytical << " absolute error: "
                << std::abs(interface_analytical - vertices_and_temperatures_all[0].first[dim - 1])
                << std::endl;
          generic_data_out.get_vector("level_set").zero_out_ghost_values();
          generic_data_out.get_vector("temperature").zero_out_ghost_values();
        }
      n_time_step += 1;
    }


  private:
    const double x_max;
    double       y_interface = 5.e-5;
    double       T_wall      = 383.15;

    // post-processing
    std::vector<Point<dim>>                                 vertices_along_vertical_axis;
    mutable std::ofstream                                   file_level_set_contour;
    std::string                                             file_name_level_set_contour;
    mutable std::ofstream                                   file_temperature_profile;
    mutable int                                             n_time_step = 0.0;
    mutable double                                          beta        = -1.0;
    mutable Utilities::MPI::RemotePointEvaluation<dim, dim> remote_point_evaluation;
    mutable bool                                            remote_point_is_initialized = false;
  };
} // namespace MeltPoolDG::Simulation::StefansProblem1WithFlowAndHeat
