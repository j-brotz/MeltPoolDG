#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>

#include "../compressible_flow_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  /**
   * @brief Exact solution function for the isentropic vortex simulation.
   */
  template <int dim, typename number>
  class IsentropicVortexExactSolution : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param time Current simulation time.
     * @param gamma Specific heat ratio.
     */
    explicit IsentropicVortexExactSolution(const number time, const number gamma)
      : dealii::Function<dim, number>(dim + 2, time)
      , gamma(gamma)
    {}

    /**
     * @brief Computes the current function value for a specific component at a given point @p x.
     *
     * @param x Given point at which the function should be evaluated.
     * @param component Component for which the function value should be returned.
     */
    number
    value(const dealii::Point<dim, number> &x, const unsigned int component) const override
    {
      AssertDimension(dim, 2);

      constexpr number           beta         = 5;
      const number               current_time = this->get_time();
      dealii::Point<dim, number> x0;
      x0[0] = 5.;
      const number radius_sqr =
        (x - x0).norm_square() - 2. * (x[0] - x0[0]) * current_time + current_time * current_time;
      const number factor = beta / (dealii::numbers::PI * 2) * std::exp(1. - radius_sqr);
      const number density_log =
        std::log2(std::abs(1. - (gamma - 1.) / gamma * 0.25 * factor * factor));
      const number density = std::exp2(density_log * (1. / (gamma - 1.)));
      const number u       = 1. - factor * (x[1] - x0[1]);
      const number v       = factor * (x[0] - current_time - x0[0]);

      if (component == 0)
        return density;
      else if (component == 1)
        return density * u;
      else if (component == 2)
        return density * v;
      else
        {
          const number pressure = std::exp2(density_log * (gamma / (gamma - 1.)));
          return pressure / (gamma - 1.) + 0.5 * (density * u * u + density * v * v);
        }
    }

  private:
    /// Specific heat ratio
    const number gamma;
  };

  /**
   * @brief A specific compressible flow simulation setup for the isentropic vortex.
   */
  template <int dim, typename number>
  class SimulationIsentropicVortex final : public ::MeltPoolDG::CompressibleFlow::Case<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param parameter_file Parameter file that contains simulation input settings.
     * @param mpi_communicator The MPI communicator used to run the simulation in parallel.
     */
    explicit SimulationIsentropicVortex(std::string parameter_file, const MPI_Comm mpi_communicator)
      : ::MeltPoolDG::CompressibleFlow::Case<dim, number>(parameter_file, mpi_communicator)
    {}

    /**
     * @brief Creates the spatial discretization for the simulation setup.
     */
    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim, number> lower_left;
      for (unsigned int d = 1; d < dim; ++d)
        lower_left[d] = -5;

      dealii::Point<dim, number> upper_right;
      upper_right[0] = 10;
      for (unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 5;

      dealii::GridGenerator::hyper_rectangle(*this->triangulation, lower_left, upper_right);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * @brief Sets the boundary conditions.
     */
    void
    set_boundary_conditions() override
    {
      auto exact_solution = std::make_shared<IsentropicVortexExactSolution<dim, number>>(
        this->parameters.time_stepping.start_time, this->parameters.material.gamma);
      this->attach_boundary_condition({0, exact_solution}, "inflow", "compressible_flow");
    }

    /**
     * @brief Sets the field functions for the simulation.
     */
    void
    set_field_conditions() override
    {
      auto exact_solution = std::make_shared<IsentropicVortexExactSolution<dim, number>>(
        this->parameters.time_stepping.start_time, this->parameters.material.gamma);
      this->attach_initial_condition(exact_solution, "compressible_flow");
      this->attach_field_function(exact_solution, "exact_solution", "compressible_flow");
    }

    /**
     * @brief Performs post-processing by evaluating and outputting error norms.
     *
     * @param generic_data_out A generic utility for managing simulation output data.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      IsentropicVortexExactSolution<dim, number> exact_solution(generic_data_out.get_time(),
                                                                this->parameters.material.gamma);

      using DataToPrint =
        typename MeltPoolDG::CompressibleFlow::Case<dim, number>::DataPostprocessorData;

      std::vector<DataToPrint> postprocessor_data_vector;

      const auto density =
        Functions::ExtractedComponentsFunction<dim, number>(exact_solution, 0, 1);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "density", .reference_function = density});

      const auto momentum =
        Functions::ExtractedComponentsFunction<dim, number>(exact_solution, 1, dim);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "momentum", .reference_function = momentum});

      const auto total_energy =
        Functions::ExtractedComponentsFunction<dim, number>(exact_solution, 1 + dim, 1);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "total energy", .reference_function = total_energy});

      this->print_relative_norm_fitted(generic_data_out, postprocessor_data_vector, "error");
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
