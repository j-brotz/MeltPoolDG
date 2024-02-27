#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

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
 * Initial interface location: 1e-6 m (we modified it to 2.e-5, otherwise the level set would range
 * into the wall)
 *
 * boiling temperature:             373.15 K
 * temperature of the heating wall: 383.15 K
 *
 * gas (vapor) phase:
 *    -- density:      1 kg/m^3
 *    -- viscosity:    0.0001 Pa/s
 *    -- conductivity: 1e-2 W/(mK)
 *    -- capacity:     1000 J/(kgK)
 *
 * liquid phase:
 *    -- density:      1 kg/m^3
 *    -- viscosity:    0.01 Pa/s
 *    -- conductivity: 1 W/(mK) (Note: thermal diffusivity of the liquid phase was increased by
 * order of magnitudes)
 *    -- capacity:     1000 J/(kgK)
 *
 * Enthalpy of evaporation: 10^6 J/kg
 * Surface tension coefficient: 0.01 N/m
 *
 * NOTE: Due to the equal densities in the two phases, no flow velocities will be induced.
 *
 * NOTE: We assumed the conductivity to be the same in the gas and the fluid phase.
 *
 * NOTE: In the publication, they did not use the evaporative mass flux calculated according to
 * Schrage's theory. At the moment, it is unclear how the mass flux is calculated. Thus, at the
 * moment a comparison with the presented analytical solution is not possible.
 *
 * We had to specify the enthalpy of evaporation as 10^2 and the evaporation coefficient of 1.
 * Otherwise, the evaporative mass flux would have been unrealistically high.
 *
 * @todo: Find a way to compare with the analytical solution.
 */

namespace MeltPoolDG::Simulation::StefansProblem1WithFlowAndHeat
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static constexpr double x_min       = 0.0;
  static constexpr double y_min       = 0.0;
  static constexpr double y_max       = 1.e-3;
  static constexpr double y_interface = 2.e-5;
  static constexpr double T_bottom    = 383.15;
  static constexpr double T_sat       = 373.15;

  template <int dim>
  class InitialValuesLS : public Function<dim>
  {
  public:
    InitialValuesLS(const double eps)
      : Function<dim>()
      , signed_distance_plane(Point<dim>::unit_vector(dim - 1) * y_interface,
                              Point<dim>::unit_vector(dim - 1))
      , eps(eps)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
        signed_distance_plane.value(p), eps);
    }

  private:
    const Functions::SignedDistance::Plane<dim> signed_distance_plane;
    const double                                eps;
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
      return T_bottom - (T_bottom - T_sat) / y_interface * p[dim - 1];
    }
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

    {
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        file_level_set_contour.open(this->parameters.output.directory + "/" +
                                    this->parameters.output.paraview.filename +
                                    "_level_set_contour_over_time");
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

      const unsigned int n_elements_per_edge =
        std::pow(dim, this->parameters.base.global_refinements);
      std::vector<unsigned int> refinements(dim, 1);

      if (dim > 1)
        refinements[dim - 1] = n_elements_per_edge;
      else
        refinements[0] = this->parameters.base.global_refinements;

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
          GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                    refinements,
                                                    bottom_left,
                                                    top_right);
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

      if (this->parameters.evapor.ls_value_liquid == 1)
        {
          // lower part = gas; upper part = liquid
          this->attach_dirichlet_boundary_condition(
            upper_bc, std::make_shared<Functions::ConstantFunction<dim>>(1.0), "level_set");
          this->attach_dirichlet_boundary_condition(
            lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1.0), "level_set");
          this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
          // no volume expansion in case of equal densities
          this->attach_no_slip_boundary_condition(upper_bc, "navier_stokes_u");
          // this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");
        }
      else
        AssertThrow(false, ExcNotImplemented());

      // no volume expansion in case of equal densities
      this->attach_fix_pressure_constant_condition(lower_bc, "navier_stokes_p");
      this->attach_fix_pressure_constant_condition(upper_bc, "navier_stokes_p");

      if (dim >= 2)
        {
          this->attach_fix_pressure_constant_condition(left_bc, "navier_stokes_p");
          this->attach_fix_pressure_constant_condition(right_bc, "navier_stokes_p");
          this->attach_symmetry_boundary_condition(left_bc, "navier_stokes_u");
          this->attach_symmetry_boundary_condition(right_bc, "navier_stokes_u");
        }

      this->attach_dirichlet_boundary_condition(lower_bc,
                                                std::make_shared<InitialValuesTemperature<dim>>(),
                                                "heat_transfer");

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
      double eps =
        UtilityFunctions::compute_initial_epsilon<dim>(this->parameters, *this->triangulation);

      AssertThrow(eps > 0, ExcNotImplemented());

      this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(eps), "level_set");
      this->attach_initial_condition(
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>(dim)), "navier_stokes_u");
      // this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(T_sat),
      //"heat_transfer");
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(),
                                     "heat_transfer");
    }

    void
    do_postprocessing([[maybe_unused]] const GenericDataOut<dim> &generic_data_out) const final
    {
      if (this->parameters.output.do_user_defined_postprocessing)
        {
          if constexpr (dim > 1)
            {
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
                  vertices_and_temperatures,
                  this->mpi_communicator,
                  [](const auto &a, const auto &b) {
                    auto result = a;
                    result.insert(result.end(), b.begin(), b.end());
                    return result;
                  });

              // write values to file
              if (file_level_set_contour.is_open())
                file_level_set_contour << generic_data_out.get_time() << " "
                                       << vertices_and_temperatures_all[0].first[dim - 1] << " "
                                       << vertices_and_temperatures_all[0].second << std::endl;
            }
        }
    }

  private:
    const double          x_max;
    mutable std::ofstream file_level_set_contour;
  };
} // namespace MeltPoolDG::Simulation::StefansProblem1WithFlowAndHeat
