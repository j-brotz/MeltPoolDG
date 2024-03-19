#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <meltpooldg/interface/simulation_base.hpp>
#include <meltpooldg/level_set/level_set_tools.hpp>
#include <meltpooldg/post_processing/slice.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


/**
 * This example is derived from
 *
 * Hardt, S., and F. Wondra. "Evaporation model for interfacial flows based on a continuum-field
 * representation of the source terms." Journal of Computational Physics 227.11 (2008): 5871-5895.
 *
 * and represents the film boiling example.
 */

namespace MeltPoolDG::Simulation::FilmBoiling
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static double factor_height = 1.0;
  static double delta_T       = 5;

  /**
   *  Create a signed distance function, which is used to compute the initial
   *  level set field.
   */
  template <int dim>
  class InitialSignedDistance : public Function<dim>
  {
  public:
    InitialSignedDistance(const double y_interface, const double lambda0)
      : Function<dim>()
      , y_interface(y_interface)
      , lambda0(lambda0)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      // compute the projected distance from the symmetry plane characterized by dim-1=0.0
      const double projected_radius = dim == 1 ? 0 :
                                      dim == 2 ? p[0] :
                                                 std::sqrt(p[0] * p[0] + p[1] * p[1]);

      const double y_interface_disturbed =
        y_interface + lambda0 / 160. *
                        std::cos(2 * numbers::PI * projected_radius /
                                 lambda0); // according to Hardt & Wondra (2007)

      Point<dim> at_interface;
      at_interface[dim - 1] = y_interface_disturbed;

      const auto plane =
        Functions::SignedDistance::Plane<dim>(at_interface, Point<dim>::unit_vector(dim - 1));

      return plane.value(p);
    }
    double y_interface, lambda0;
  };
  /**
   *  Initial temperature field
   */
  template <int dim>
  class InitialValuesTemperature : public Function<dim>
  {
  public:
    InitialValuesTemperature(const double T_max,
                             const double T_min,
                             const double y_interface,
                             const double lambda0)
      : Function<dim>()
      , T_max(T_max)
      , T_min(T_min)
      , y_interface(y_interface)
      , lambda0(lambda0)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      double y_interface_disturbed =
        y_interface + lambda0 / 160. * std::cos(2 * numbers::PI * p[0] / lambda0);

      if (p[dim - 1] > y_interface_disturbed)
        return T_min;
      else
        return T_max - (T_max - T_min) / y_interface_disturbed * p[dim - 1];
    }
    const double T_max, T_min, y_interface, lambda0;
  };

  /**
   *      This class collects all relevant input data for the simulation.
   */
  template <int dim>
  class SimulationFilmBoiling : public SimulationBase<dim>
  {
  public:
    SimulationFilmBoiling(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
      , lambda0(
          2. * numbers::PI *
          std::sqrt(3. * this->parameters.flow.surface_tension.surface_tension_coefficient /
                    (this->parameters.flow.gravity * (this->parameters.material.liquid.density -
                                                      this->parameters.material.gas.density))))
      , x_max(lambda0 / 2.)
      , y_max(lambda0)
      , x_min(-x_max)
      , tria_slice(mpi_communicator)
    {}

    void
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter("factor height",
                          factor_height,
                          "Set a factor multiplied with the height.");
        prm.add_parameter("delta T",
                          delta_T,
                          "Set the delta of the wall temperature to the boiling temperature.");
      }
      prm.leave_subsection();
    }

    void
    create_spatial_discretization() override
    {
      y_max *= factor_height;

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

      // create mesh
      std::vector<unsigned int> subdivisions(
        dim,
        5 * (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP ?
               Utilities::pow(2, this->parameters.base.global_refinements) :
               1));
      // Create elements with a width to height ratio of 3. This leads to a higher resolution
      // in the dim-1 direction, which is the predominant direction of this example.
      subdivisions[dim - 1] *= 3;

      if (this->parameters.base.fe.type == FiniteElementType::FE_SimplexP)
        GridGenerator::subdivided_hyper_rectangle_with_simplices(
          *this->triangulation, subdivisions, bottom_left, top_right, true /*colorize*/);
      else
        GridGenerator::subdivided_hyper_rectangle(
          *this->triangulation, subdivisions, bottom_left, top_right, true /*colorize*/);

      // create slice for postprocessing
      if constexpr (dim == 3)
        {
          GridGenerator::subdivided_hyper_rectangle(tria_slice,
                                                    {1, 3} /*subdivisions*/,
                                                    Point<2>(x_min, y_min),
                                                    Point<2>(x_max, y_max));

          GridTools::rotate(Point<3>::unit_vector(0), 0.5 * numbers::PI, tria_slice);

          tria_slice.refine_global(4);
        }
    }

    void
    set_boundary_conditions() final
    {
      // face numbering according to the deal.II colorize flag
      const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      this->attach_no_slip_boundary_condition(lower_bc, "navier_stokes_u");
      this->attach_open_boundary_condition(upper_bc, "navier_stokes_u");
      this->attach_dirichlet_boundary_condition(lower_bc,
                                                std::make_shared<Functions::ConstantFunction<dim>>(
                                                  this->parameters.material.boiling_temperature +
                                                  5.),
                                                "heat_transfer");

      // @note: this BC is necessary
      this->attach_dirichlet_boundary_condition(
        lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1), "level_set");

      if (dim > 1)
        this->attach_periodic_boundary_condition(left_bc, right_bc, 0);

      if (dim > 2)
        this->attach_periodic_boundary_condition(front_bc, back_bc, 1);

      if (this->parameters.base.fe.type != FiniteElementType::FE_SimplexP)
        this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(
        std::make_shared<InitialSignedDistance<dim>>(9. * lambda0 / 128., lambda0),
        "signed_distance");

      this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                     "navier_stokes_u");

      // assign initial temperature such that at the interface the boiling temperature is met
      this->attach_initial_condition(std::make_shared<InitialValuesTemperature<dim>>(
                                       this->parameters.material.boiling_temperature + delta_T,
                                       this->parameters.material.boiling_temperature,
                                       9. * lambda0 / 128.,
                                       lambda0),
                                     "heat_transfer");
    }

    void
    do_postprocessing([[maybe_unused]] const GenericDataOut<dim> &generic_data_out) const final
    {
      if (this->parameters.output.do_user_defined_postprocessing == false)
        return;

      // create slice
      if constexpr (dim == 3)
        {
          if (!slice)
            {
              const std::vector<std::string> req_vars = {"level_set", "temperature", "velocity"};

              slice = std::make_shared<PostProcessingTools::SliceCreator<3>>(
                generic_data_out, tria_slice, req_vars, this->parameters.output);
            }
          // @todo: We need to reinit, since generic_data_out is currently created
          // for every time step.
          slice->reinit(generic_data_out);

          slice->process(n_written_time_step);
          ++n_written_time_step;
        }

      // create iso-surface
      if constexpr (dim >= 2)
        {
          Triangulation<dim - 1, dim> tria;

          GridGenerator::create_triangulation_with_marching_cube_algorithm(
            tria,
            generic_data_out.get_mapping(),
            generic_data_out.get_dof_handler("level_set"),
            generic_data_out.get_vector("level_set"),
            0 /*iso_level*/,
            1 /*n_subdivisions*/);


          DataOut<dim - 1, dim> data_out;
          data_out.attach_triangulation(tria);
          if (tria.n_cells() > 0)
            data_out.build_patches();
          data_out.write_vtu_with_pvtu_record(
            "./", "data_out_02" /*TODO*/, 0, this->mpi_communicator, 1 /*TODO*/, 1);
        }
    }

  private:
    const double                               lambda0 = 0.0;
    const double                               x_max;
    double                                     y_max;
    const double                               x_min;
    const double                               y_min               = 0.0;
    mutable unsigned int                       n_written_time_step = 0;
    parallel::distributed::Triangulation<2, 3> tria_slice;

    mutable std::shared_ptr<PostProcessingTools::SliceCreator<3>> slice;
  };
} // namespace MeltPoolDG::Simulation::FilmBoiling
