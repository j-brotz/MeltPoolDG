#include "unidirectional_heat_transfer.hpp"
//

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/core/case_registration.hpp>
#include <meltpooldg/heat/heat_data.hpp>
#include <meltpooldg/utilities/boundary_ids_colorized.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <vector>


namespace MeltPoolDG::Simulation::UnidirectionalHeatTransfer
{
  using namespace dealii;

  static constexpr double x_min = 0.0;
  static constexpr double x_max = 0.1;

  template <int dim, typename number>
  class LinearTemp : public Function<dim, number>
  {
  public:
    LinearTemp(const number left_temperature = 0.0, const number right_temperature = 0.1)
      : Function<dim, number>(1, 0)
      , left_temp(left_temperature)
      , right_temp(right_temperature)
    {}

    number
    value(const Point<dim, number> &p, const unsigned int /*component*/) const override
    {
      return left_temp + (right_temp - left_temp) * (p[0] - x_min) / (x_max - x_min);
    }

  private:
    const number left_temp;
    const number right_temp;
  };

  template <int dim, typename number>
  class UnidirectionalVelocityField : public Function<dim, number>
  {
  public:
    UnidirectionalVelocityField(number velocity)
      : Function<dim, number>(dim)
      , vel(velocity)
    {}

    number
    value(const Point<dim, number> &, const unsigned int component) const override
    {
      if (component == 0)
        return -vel;
      else
        return 0.0;
    }

  private:
    const number vel;
  };

  template <int dim, typename number>
  class HorizontalLevelSet : public Function<dim, number>
  {
  public:
    HorizontalLevelSet(const bool do_heaviside)
      : Function<dim, number>(1)
      , heaviside(do_heaviside)
    {}

    number
    value(const Point<dim, number> &p, const unsigned int) const override
    {
      const auto signed_distance = level - p[1];
      if (heaviside)
        return UtilityFunctions::CharacteristicFunctions::heaviside(signed_distance, eps);
      else
        return signed_distance;
    }

  private:
    const bool   heaviside;
    const number eps   = 0.01;
    const number level = x_max / 2;
  };

  template <int dim, typename number>
  class CovectedVerticalLevelSetHeaviside : public Function<dim, number>
  {
  public:
    CovectedVerticalLevelSetHeaviside(const number velocity, const bool do_heaviside)
      : Function<dim, number>(1)
      , velocity(velocity)
      , heaviside(do_heaviside)
    {}

    number
    value(const Point<dim, number> &p, const unsigned int) const override
    {
      const auto signed_distance = level - p[0] - velocity * this->get_time();

      if (heaviside)
        return UtilityFunctions::CharacteristicFunctions::heaviside(signed_distance, eps);
      else
        return signed_distance;
    }

  private:
    const number eps   = 0.01;
    const number level = x_max * 2. / 3.;
    const number velocity;
    const bool   heaviside;
  };


  template <int dim, typename number>
  SimulationUnidirectionalHeatTransfer<dim, number>::SimulationUnidirectionalHeatTransfer(
    std::string    parameter_file,
    const MPI_Comm mpi_communicator)
    : Heat::HeatTransferCase<dim, number>(parameter_file, mpi_communicator)
  {}

  template <int dim, typename number>
  bool
  SimulationUnidirectionalHeatTransfer<dim, number>::add_simulation_specific_parameters(
    dealii::ParameterHandler &prm)
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


  template <int dim, typename number>
  void
  SimulationUnidirectionalHeatTransfer<dim, number>::create_spatial_discretization()
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


  template <int dim, typename number>
  void
  SimulationUnidirectionalHeatTransfer<dim, number>::set_boundary_conditions()
  {
    // face numbering according to the deal.II colorize flag
    [[maybe_unused]] const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
      get_colorized_rectangle_boundary_ids<dim>();

    if (this->parameters.heat.radiation.emissivity > 0.0)
      this->attach_boundary_condition(dim == 1 ? lower_bc : right_bc, "radiation", "heat_transfer");

    if (this->parameters.heat.convection.convection_coefficient > 0.0)
      this->attach_boundary_condition(dim == 1 ? lower_bc : right_bc,
                                      "convection",
                                      "heat_transfer");
  }


  template <int dim, typename number>
  void
  SimulationUnidirectionalHeatTransfer<dim, number>::set_field_conditions()
  {
    if (do_solidification)
      this->attach_initial_condition(std::make_shared<LinearTemp<dim, number>>(1960.0, 1980.0),
                                     "heat_transfer");
    else if (this->parameters.heat.radiation.emissivity > 0.0 ||
             this->parameters.heat.convection.convection_coefficient > 0.0)
      this->attach_initial_condition(
        std::make_shared<Functions::ConstantFunction<dim, number>>(1000), "heat_transfer");
    else if (this->parameters.evapor.evaporative_cooling.enable)
      this->attach_initial_condition(std::make_shared<LinearTemp<dim, number>>(1960.0, 2040.0),
                                     "heat_transfer");
    else
      this->attach_initial_condition(std::make_shared<LinearTemp<dim, number>>(), "heat_transfer");

    if (velocity != 0.0)
      this->attach_field_function(std::make_shared<UnidirectionalVelocityField<dim, number>>(
                                    velocity),
                                  "prescribed_velocity",
                                  "heat_transfer");

    if (do_two_phase)
      {
        if (this->parameters.heat.operator_type == Heat::TwoPhaseOperatorType::diffuse)
          {
            if (!do_solidification)
              this->attach_initial_condition(std::make_shared<HorizontalLevelSet<dim, number>>(
                                               true /* do_heaviside */),
                                             "prescribed_heaviside");
            else
              this->attach_initial_condition(
                std::make_shared<CovectedVerticalLevelSetHeaviside<dim, number>>(
                  velocity, true /* do_heaviside */),
                "prescribed_heaviside");
          }
        else if (this->parameters.heat.operator_type == Heat::TwoPhaseOperatorType::cut)
          {
            if (velocity == 0.0)
              this->attach_initial_condition(std::make_shared<HorizontalLevelSet<dim, number>>(
                                               false /* do_heaviside */),
                                             "prescribed_signed_distance");
            else
              this->attach_initial_condition(
                std::make_shared<CovectedVerticalLevelSetHeaviside<dim, number>>(
                  velocity, false /* do_heaviside */),
                "prescribed_signed_distance");
          }
      }
  }

  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase,
                           SimulationUnidirectionalHeatTransfer,
                           "unidirectional_heat_transfer",
                           1,
                           double);
  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase,
                           SimulationUnidirectionalHeatTransfer,
                           "unidirectional_heat_transfer",
                           2,
                           double);
  MELTPOOLDG_REGISTER_CASE(Heat::HeatTransferCase,
                           SimulationUnidirectionalHeatTransfer,
                           "unidirectional_heat_transfer",
                           3,
                           double);
} // namespace MeltPoolDG::Simulation::UnidirectionalHeatTransfer
