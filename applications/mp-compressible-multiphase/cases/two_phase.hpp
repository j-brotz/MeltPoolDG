/**
 * @brief Simulation of (currently) one-dimensional two-phase flows with one phase interface.
 * The initial conditions and boundary types are used from the user input file.
 *
 *       _____________________________________
 *    ->|      liquid      |        gas      |->
 *     x=-5e-4           x=0.013e-4         x=5e-4
 */

#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/boundary_ids_colorized.hpp>

#include "../compressible_multiphase_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleMultiphase
{
  /**
   * @brief Initial flow field function.
   */
  template <int dim, typename number>
  class InitialField : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param ic_gas_phase Function for the initial conditions of the gas phase.
     * @param ic_liquid_phase Function for the initial conditions of the liquid phase.
     * @param gas_phase_is_first Indicator whether the gas phase is the first phase in the two-phase
     * system.
     *
     * @throws dealii::ExcNotImplemented If `dim != 1`.
     */
    explicit InitialField(std::string ic_gas_phase,
                          std::string ic_liquid_phase,
                          const bool  gas_phase_is_first = true)
      : dealii::Function<dim, number>(2 * (dim + 2))
      , gas_phase_is_first(gas_phase_is_first)
    {
      parsing_function_gas->initialize(dim == 3 ?
                                         std::string("x,y,z") :
                                         (dim == 2 ? std::string("x,y") : std::string("x")),
                                       ic_gas_phase,
                                       constants,
                                       false);
      parsing_function_liquid->initialize(dim == 3 ?
                                            std::string("x,y,z") :
                                            (dim == 2 ? std::string("x,y") : std::string("x")),
                                          ic_liquid_phase,
                                          constants,
                                          false);

      // Currently, only 1D simulations are possible
      Assert(dim == 1, dealii::ExcNotImplemented());
    }

    /**
     * @brief Computes the current function value for a specific @p component at a given point @p p.
     *
     * @param p Point at which the function should be evaluated.
     * @param component Component for which the function value should be returned.
     */
    number
    value(const dealii::Point<dim, number> &p, const unsigned int component) const final
    {
      std::array<unsigned int, dim + 2> gas_components;
      std::array<unsigned int, dim + 2> liquid_components;

      for (unsigned int i = 0; i < dim + 2; ++i)
        {
          if (gas_phase_is_first)
            {
              gas_components[i]    = i;
              liquid_components[i] = i + dim + 2;
            }
          else
            {
              gas_components[i]    = i + dim + 2;
              liquid_components[i] = i;
            }
        }

      // gas phase
      if (component == gas_components[0])
        return parsing_function_gas->value(p, 0);
      else if (component == gas_components[1])
        return parsing_function_gas->value(p, 1);
      else if (component == gas_components[2])
        return parsing_function_gas->value(p, 2);

      // liquid phase
      else if (component == liquid_components[0])
        return parsing_function_liquid->value(p, 0);
      else if (component == liquid_components[1])
        return parsing_function_liquid->value(p, 1);
      else if (component == liquid_components[2])
        return parsing_function_liquid->value(p, 2);
      else
        return 0.;
    }

  private:
    /// Indicator whether the gas phase is the first phase in the two-phase system
    bool gas_phase_is_first;

    /// Constants for function parsing
    std::map<std::string, number> constants;

    /// Function parser for initial conditions in gas phase
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_gas =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);

    /// Function parser for initial conditions in liquid phase
    std::unique_ptr<dealii::FunctionParser<dim>> parsing_function_liquid =
      std::make_unique<dealii::FunctionParser<dim>>(dim + 2);
  };

  /**
   * @brief A specific compressible flow simulation setup for a one-dimensional two-phase case with
   * one (moving) interface.
   */
  template <int dim, typename number>
  class SimulationTwoPhase final : public Multiphase::CompressibleMultiphaseCase<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param parameter_file Parameter file that contains simulation input settings.
     * @param mpi_communicator The MPI communicator used to run the simulation in parallel.
     */
    explicit SimulationTwoPhase(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Multiphase::CompressibleMultiphaseCase<dim, number>(parameter_file, mpi_communicator)
    {}

    /**
     * @brief Creates the spatial discretization for the simulation setup.
     */
    void
    create_spatial_discretization() override
    {
      // TODO: no distributed triangulation possible for dim=1
      this->triangulation =
        std::make_shared<dealii::parallel::shared::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim, number> lower_left;
      lower_left[0] = -5.e-4;
      for (unsigned int d = 1; d < dim; ++d)
        lower_left[d] = 0.;

      dealii::Point<dim, number> upper_right;
      upper_right[0] = 5.e-4;
      for (unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 0.;

      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 20;
      for (unsigned int d = 1; d < dim; ++d)
        subdivisions[d] = 1;

      dealii::GridGenerator::subdivided_hyper_rectangle(
        *this->triangulation, subdivisions, lower_left, upper_right, true);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * @brief Sets the boundary conditions.
     */
    void
    set_boundary_conditions() override
    {
      auto inflow_outflow_solution =
        std::make_shared<InitialField<dim, number>>(initial_conditions.gas,
                                                    initial_conditions.liquid);

      // face numbering according to the deal.II colorize flag
      const auto [lower_bc, upper_bc, left_bc, right_bc, front_bc, back_bc] =
        get_colorized_rectangle_boundary_ids<dim>();

      // left boundary
      this->attach_boundary_condition({lower_bc, inflow_outflow_solution},
                                      left_boundary_condition,
                                      "compressible_multiphase_flow");
      // right boundary
      this->attach_boundary_condition({upper_bc, inflow_outflow_solution},
                                      right_boundary_condition,
                                      "compressible_multiphase_flow");
    }

    /**
     * @brief Sets the field functions for the simulation.
     */
    void
    set_field_conditions() override
    {
      // The solution vector is ordered, such that the liquid phase is the first phase and the gas
      // phase is the second phase.
      auto initial_condition =
        std::make_shared<InitialField<dim, number>>(initial_conditions.gas,
                                                    initial_conditions.liquid,
                                                    false /*gas_phase_is_first*/);
      this->attach_initial_condition(initial_condition, "compressible_multiphase_flow");

      // set level-set function
      dealii::Point<dim, number> p;
      // avoid phase interface colliding with element face (bug in dealii has to be fixed)
      p[0] = 0.0135468746e-4;

      dealii::Tensor<1, dim, number> normal;
      normal[0] = -1.;

      const auto level_set =
        std::make_shared<dealii::Functions::SignedDistance::Plane<dim>>(p, normal);
      this->attach_field_function(level_set, "level_set", "compressible_multiphase_flow");
    }

    /**
     * @brief Performs post-processing by evaluating and outputting error norms.
     *
     * @param generic_data_out A generic utility for managing simulation output data.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      InitialField<dim, number> reference_values(initial_conditions.gas, initial_conditions.liquid);
      this->print_relative_norm(generic_data_out, reference_values, "error");
    }

    /**
     * @brief Add simulation specific parameters to the parameter handler.
     *
     * @param prm The parameter handler to which the parameters are added.
     */
    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      const std::string boundary_condition_types_doc =
        "The following boundary conditions are supported:\n"
        "\t 'no_slip_wall', 'slip_wall', 'inflow', 'outflow_fixed_energy', "
        "\t 'outflow_fixed_pressure'";

      prm.enter_subsection("simulation specific parameters");
      {
        prm.add_parameter("left boundary condition",
                          left_boundary_condition,
                          boundary_condition_types_doc);
        prm.add_parameter("right boundary condition",
                          right_boundary_condition,
                          boundary_condition_types_doc);
        prm.enter_subsection("initial conditions");
        {
          prm.add_parameter("gas",
                            initial_conditions.gas,
                            "Initial condition function for gas phase.");
          prm.add_parameter("liquid",
                            initial_conditions.liquid,
                            "Initial condition function for liquid phase.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Boundary conditions, the options are:
    /// "no_slip_wall", "slip_wall", "inflow", "outflow_fixed_energy", "outflow_fixed_pressure"
    std::string left_boundary_condition  = "no_slip_wall";
    std::string right_boundary_condition = "no_slip_wall";

    /// Functions for the initial conditions
    struct InitialConditions
    {
      std::string gas{};
      std::string liquid{};
    } initial_conditions;
  };
} // namespace MeltPoolDG::Simulation::CompressibleMultiphase
