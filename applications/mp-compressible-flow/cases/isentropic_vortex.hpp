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

  template <int dim>
  class IsentropicVortexExactSolution : public Function<dim>
  {
  public:
    IsentropicVortexExactSolution(const double time, const double gamma)
      : Function<dim>(dim + 2, time)
      , gamma(gamma)
    {}

    double
    value(const Point<dim> &x, const unsigned int component) const override
    {
      AssertDimension(dim, 2);

      constexpr double beta         = 5;
      const double     current_time = this->get_time();
      Point<dim>       x0;
      x0[0] = 5.;
      const double radius_sqr =
        (x - x0).norm_square() - 2. * (x[0] - x0[0]) * current_time + current_time * current_time;
      const double factor = beta / (numbers::PI * 2) * std::exp(1. - radius_sqr);
      const double density_log =
        std::log2(std::abs(1. - (gamma - 1.) / gamma * 0.25 * factor * factor));
      const double density = std::exp2(density_log * (1. / (gamma - 1.)));
      const double u       = 1. - factor * (x[1] - x0[1]);
      const double v       = factor * (x[0] - current_time - x0[0]);

      if (component == 0)
        return density;
      else if (component == 1)
        return density * u;
      else if (component == 2)
        return density * v;
      else
        {
          const double pressure = std::exp2(density_log * (gamma / (gamma - 1.)));
          return pressure / (gamma - 1.) + 0.5 * (density * u * u + density * v * v);
        }
    }

  private:
    const double gamma;
  };

  template <int dim>
  class SimulationIsentropicVortex final : public Flow::CompressibleFlowCase<dim>
  {
  public:
    SimulationIsentropicVortex(std::string parameter_file, const MPI_Comm mpi_communicator)
      : Flow::CompressibleFlowCase<dim>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      this->triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      Point<dim> lower_left;
      for (unsigned int d = 1; d < dim; ++d)
        lower_left[d] = -5;

      Point<dim> upper_right;
      upper_right[0] = 10;
      for (unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 5;

      GridGenerator::hyper_rectangle(*this->triangulation, lower_left, upper_right);
      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      auto exact_solution = std::make_shared<IsentropicVortexExactSolution<dim>>(
        this->parameters.time_stepping.start_time,
        this->parameters.flow.material_data_gas_phase.gamma);
      this->attach_boundary_condition({0, exact_solution}, "inflow", "compressible_flow");
    }

    void
    set_field_conditions() override
    {
      auto exact_solution = std::make_shared<IsentropicVortexExactSolution<dim>>(
        this->parameters.time_stepping.start_time,
        this->parameters.flow.material_data_gas_phase.gamma);
      this->attach_initial_condition(exact_solution, "compressible_flow");
      this->attach_field_function(exact_solution, "exact_solution", "compressible_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim, double> &generic_data_out) const override
    {
      IsentropicVortexExactSolution<dim> exact_solution(
        generic_data_out.get_time(), this->parameters.flow.material_data_gas_phase.gamma);
      this->print_relative_norm(generic_data_out, exact_solution, "Error");
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
