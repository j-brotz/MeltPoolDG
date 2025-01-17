#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/parameters.hpp>
#include <meltpooldg/core/simulation_base.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

/**
 * This example represents a simple test example for heat transfer.
 * It offers the following configurations dependent on the input parameters:
 *
 *  case 1: emissivity > 0, convection_coefficient = 0
 *             adiabatic
 *            +--------+
 *            |        |
 * adiabatic  |        |   radiation
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 2: emissivity = 0, convection_coefficient > 0
 *             adiabatic
 *            +--------+
 *            |        |
 * adiabatic  |        |   convection
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 3: emissivity > 0, convection_coefficient > 0
 *             adiabatic
 *            +--------+
 *            |        |
 * adiabatic  |        |   radiation + convection
 *            |        |
 *            +--------+
 *            adiabatic
 *
 *  case 4: emissivity = 0, convection_coefficient = 0
 *
 *             adiabatic
 *            +--------+
 *            |        |
 *  adiabatic | T0=lin |   adiabatic
 *            |        |
 *            +--------+
 *            adiabatic
 *
 */

namespace MeltPoolDG::Simulation::UnidirectionalHeatTransfer
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;

  static constexpr double x_min = 0.0;
  static constexpr double x_max = 0.1;

  /*
   *      This class collects all relevant input data for the level set simulation
   */
  template <int dim>
  class LinearTemp : public Function<dim>
  {
  public:
    LinearTemp(const double left_temperature = 0.0, const double right_temperature = 0.1)
      : Function<dim>(1, 0)
      , left_temp(left_temperature)
      , right_temp(right_temperature)
    {}

    double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return left_temp + (right_temp - left_temp) * (p[0] - x_min) / (x_max - x_min);
    }

  private:
    const double left_temp;
    const double right_temp;
  };

  template <int dim>
  class UnidirectionalVelocityField : public Function<dim>
  {
  public:
    UnidirectionalVelocityField(double velocity)
      : Function<dim>(dim)
      , vel(velocity)
    {}

    double
    value(const Point<dim> &, const unsigned int component) const override
    {
      if (component == 0)
        return -vel;
      else
        return 0.0;
    }

  private:
    const double vel;
  };

  template <int dim>
  class HorizontalLevelSet : public Function<dim>
  {
  public:
    HorizontalLevelSet(const bool do_heaviside)
      : Function<dim>(1)
      , heaviside(do_heaviside)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      const auto signed_distance = level - p[1];
      if (heaviside)
        return UtilityFunctions::CharacteristicFunctions::heaviside(signed_distance, eps);
      else
        return signed_distance;
    }

  private:
    const bool   heaviside;
    const double eps   = 0.01;
    const double level = x_max / 2;
  };

  template <int dim>
  class CovectedVerticalLevelSetHeaviside : public Function<dim>
  {
  public:
    CovectedVerticalLevelSetHeaviside(const double velocity, const bool do_heaviside)
      : Function<dim>(1)
      , velocity(velocity)
      , heaviside(do_heaviside)
    {}

    double
    value(const Point<dim> &p, const unsigned int) const override
    {
      const auto signed_distance = level - p[0] - velocity * this->get_time();

      if (heaviside)
        return UtilityFunctions::CharacteristicFunctions::heaviside(signed_distance, eps);
      else
        return signed_distance;
    }

  private:
    const double eps   = 0.01;
    const double level = x_max * 2. / 3.;
    const double velocity;
    const bool   heaviside;
  };

  template <int dim>
  class SimulationUnidirectionalHeatTransfer : public MeltPoolCase<dim>
  {
  private:
    bool   do_solidification = false;
    bool   do_two_phase      = false;
    double velocity          = 0.0;

  public:
    SimulationUnidirectionalHeatTransfer(std::string    parameter_file,
                                         const MPI_Comm mpi_communicator)
      : MeltPoolCase<dim>(parameter_file, mpi_communicator)
    {}

    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("simulation specific");
      {
        prm.add_parameter(
          "do solidification",
          do_solidification,
          "Set this parameter to true for the case to consider melting/solidification effects.");
        prm.add_parameter("do two phase",
                          do_two_phase,
                          "Set this parameter to true for the case to consider two phases.");
        prm.add_parameter("velocity", velocity, "Velocity.");
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

    void
    create_spatial_discretization() override
    {
      if constexpr (dim == 1)
        {
          AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
          this->triangulation = std::make_shared<Triangulation<dim>>();
          // create mesh
          const Point<1> left(x_min);
          const Point<1> right(x_max);
          GridGenerator::hyper_rectangle(*this->triangulation, left, right, true /*colorize*/);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }
      else if constexpr (dim == 2)
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
          // create mesh
          const Point<2> left(x_min, 0.0);
          const Point<2> right(x_max, 0.1);

          if (this->parameters.heat.operator_type == Heat::TwoPhaseOperatorType::cut)
            {
              // if we use the cut operator, make sure the number of element per direction is odd
              const std::vector<unsigned int> cell_repetitions(
                dim, Utilities::pow(2, this->parameters.base.global_refinements) | 1);
              GridGenerator::subdivided_hyper_rectangle(
                *this->triangulation, cell_repetitions, left, right, true /*colorize*/);
            }
          else
            {
              GridGenerator::hyper_rectangle(*this->triangulation, left, right, true /*colorize*/);
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
      // face numbering according to the deal.II colorize flag
      [[maybe_unused]] const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      if (this->parameters.heat.radiation.emissivity > 0.0)
        this->attach_boundary_condition(dim == 1 ? lower_bc : right_bc,
                                        "radiation",
                                        "heat_transfer");

      if (this->parameters.heat.convection.convection_coefficient > 0.0)
        this->attach_boundary_condition(dim == 1 ? lower_bc : right_bc,
                                        "convection",
                                        "heat_transfer");
    }

    void
    set_field_conditions() final
    {
      if (do_solidification)
        this->attach_initial_condition(std::make_shared<LinearTemp<dim>>(1960.0, 1980.0),
                                       "heat_transfer");
      else if (this->parameters.heat.radiation.emissivity > 0.0 ||
               this->parameters.heat.convection.convection_coefficient > 0.0)
        this->attach_initial_condition(std::make_shared<Functions::ConstantFunction<dim>>(1000),
                                       "heat_transfer");
      else if (this->parameters.evapor.evaporative_cooling.enable)
        this->attach_initial_condition(std::make_shared<LinearTemp<dim>>(1960.0, 2040.0),
                                       "heat_transfer");
      else
        this->attach_initial_condition(std::make_shared<LinearTemp<dim>>(), "heat_transfer");

      if (velocity != 0.0)
        this->attach_field_function(std::make_shared<UnidirectionalVelocityField<dim>>(velocity),
                                    "prescribed_velocity",
                                    "heat_transfer");

      if (do_two_phase)
        {
          if (this->parameters.heat.operator_type == Heat::TwoPhaseOperatorType::diffuse)
            {
              if (!do_solidification)
                this->attach_initial_condition(std::make_shared<HorizontalLevelSet<dim>>(
                                                 true /* do_heaviside */),
                                               "prescribed_heaviside");
              else
                this->attach_initial_condition(
                  std::make_shared<CovectedVerticalLevelSetHeaviside<dim>>(velocity,
                                                                           true /* do_heaviside */),
                  "prescribed_heaviside");
            }
          else if (this->parameters.heat.operator_type == Heat::TwoPhaseOperatorType::cut)
            {
              if (velocity == 0.0)
                this->attach_initial_condition(std::make_shared<HorizontalLevelSet<dim>>(
                                                 false /* do_heaviside */),
                                               "prescribed_signed_distance");
              else
                this->attach_initial_condition(
                  std::make_shared<CovectedVerticalLevelSetHeaviside<dim>>(
                    velocity, false /* do_heaviside */),
                  "prescribed_signed_distance");
            }
        }
    }
  };
} // namespace MeltPoolDG::Simulation::UnidirectionalHeatTransfer
