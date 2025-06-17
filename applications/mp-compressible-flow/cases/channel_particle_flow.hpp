#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include "../compressible_flow_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  /**
   * @brief Inflow field function.
   */
  template <int dim, typename number>
  class InflowFlowField : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param time Current simulation time.
     */
    explicit InflowFlowField(const number time)
      : dealii::Function<dim, number>(dim + 2, time)
    {}

    /**
     * @brief Computes the current function value for a specific @p component.
     *
     * @param component Component for which the function value should be returned.
     */
    number
    value(const dealii::Point<dim, number> &, const unsigned int component) const final
    {
      if (component == 0)
        return 1.;
      else if (component == 1)
        return 0.4;
      else if (component == dim + 1)
        return 3.097857142857143;
      else
        return 0.;
    }
  };

  /**
   * @brief A specific compressible flow simulation setup for channel flow with particles.
   */
  template <int dim, typename number>
  class SimulationChannelParticleFlow final : public Flow::CompressibleFlowCase<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param parameter_file Parameter file that contains simulation input settings.
     * @param mpi_communicator The MPI communicator used to run the simulation in parallel.
     *
     * @throw dealii::ExcMessage if `dim <= 1`, since the channel simulation requires at least 2D.
     */
    SimulationChannelParticleFlow(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Flow::CompressibleFlowCase<dim, number>(parameter_file, mpi_communicator)
    {
      AssertThrow(
        dim > 1,
        dealii::ExcMessage(
          "The cylinder in channel flow case requires dim > 1 but the dimension is set to dim = " +
          std::to_string(dim) + "."));
    }

    /**
     * @brief Creates the spatial discretization for the simulation setup.
     */
    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::GridGenerator::channel_with_cylinder(*this->triangulation, 0.03, 1, 0, true);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * @brief Sets the boundary conditions.
     */
    void
    set_boundary_conditions() override
    {
      auto inflow_outflow_solution =
        std::make_shared<InflowFlowField<dim, number>>(this->parameters.time_stepping.start_time);
      auto dummy_solution =
        std::make_shared<InflowFlowField<dim, number>>(this->parameters.time_stepping.start_time);
      this->attach_boundary_condition({0, inflow_outflow_solution}, "inflow", "compressible_flow");
      this->attach_boundary_condition({1, inflow_outflow_solution},
                                      "outflow_fixed_energy",
                                      "compressible_flow");
      // particle surface
      this->attach_boundary_condition({2, dummy_solution}, "no_slip_wall", "compressible_flow");
      // channel boundary
      this->attach_boundary_condition({3, dummy_solution}, "slip_wall", "compressible_flow");
    }

    /**
     * @brief Sets the field functions for the simulation.
     */
    void
    set_field_conditions() override
    {
      auto initial_condition =
        std::make_shared<InflowFlowField<dim, number>>(this->parameters.time_stepping.start_time);
      this->attach_initial_condition(initial_condition, "compressible_flow");
    }

    /**
     * @brief Performs post-processing by evaluating and outputting error norms.
     *
     * @param generic_data_out A generic utility for managing simulation output data.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      InflowFlowField<dim, number> reference_values(generic_data_out.get_time());
      this->print_relative_norm(generic_data_out, reference_values, "norm");
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
