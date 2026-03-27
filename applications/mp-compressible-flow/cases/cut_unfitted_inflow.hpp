#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/utilities/utility_functions.hpp>

#include "../compressible_flow_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  /**
   * @brief Inflow field function.
   */
  template <int dim, typename number>
  class SteadyInflowField : public dealii::Function<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @throw dealii::ExcNotImplemented Thrown if `dim` is not 2 or 3.
     */
    explicit SteadyInflowField()
      : dealii::Function<dim, number>(dim + 2)
    {
      Assert(dim == 2 or dim == 3, dealii::ExcNotImplemented());
    }

    /**
     * @brief Computes the current function value for a specific @p component.
     *
     * @param component Component for which the function value should be returned.
     */
    number
    value(const dealii::Point<dim, number> &, const unsigned int component) const final
    {
      if (component == 0)
        return 0.001;
      else if (component == 1)
        return 0.01;
      else if (component == dim + 1)
        return energy;
      else
        return 0.;
    }

    static constexpr number energy = 194.5142;
  };

  /**
   * @brief A specific compressible flow simulation setup for a flow which contains an unfitted
   * inflow boundary using cutDG.
   */
  template <int dim, typename number>
  class SimulationCutUnfittedInflow final
    : public ::MeltPoolDG::CompressibleFlow::CompressibleFlowCase<dim, number>
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param parameter_file Parameter file that contains simulation input settings.
     * @param mpi_communicator The MPI communicator used to run the simulation in parallel.
     */
    explicit SimulationCutUnfittedInflow(std::string    parameter_file,
                                         const MPI_Comm mpi_communicator)
      : ::MeltPoolDG::CompressibleFlow::CompressibleFlowCase<dim, number>(parameter_file,
                                                                          mpi_communicator)
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
        lower_left[d] = 0.;

      dealii::Point<dim, number> upper_right;
      upper_right[0] = 1.0;
      for (unsigned int d = 1; d < dim; ++d)
        upper_right[d] = 0.008;

      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 125;
      for (unsigned int d = 1; d < dim; ++d)
        subdivisions[d] = 1;

      dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        subdivisions,
                                                        lower_left,
                                                        upper_right);

      set_fitted_boundary_id(*this->triangulation);

      this->triangulation->refine_global(this->parameters.base.global_refinements);
    }

    /**
     * @brief Sets the boundary conditions.
     */
    void
    set_boundary_conditions() override
    {
      // fitted boundaries
      auto outflow_solution = std::make_shared<dealii::Functions::ConstantFunction<dim>>(
        SteadyInflowField<dim, number>::energy);
      auto dummy_solution = std::make_shared<SteadyInflowField<dim, number>>();
      this->attach_boundary_condition({BoundaryID::subsonic_outflow_with_fixed_energy,
                                       outflow_solution},
                                      "outflow_fixed_energy",
                                      "compressible_flow");
      this->attach_boundary_condition({BoundaryID::slip_wall, dummy_solution},
                                      "slip_wall",
                                      "compressible_flow");
    }

    /**
     * @brief Sets the field functions for the simulation.
     */
    void
    set_field_conditions() override
    {
      auto initial_condition = std::make_shared<SteadyInflowField<dim, number>>();
      this->attach_initial_condition(initial_condition, "compressible_flow");

      // set level-set function
      dealii::Point<dim, number> center;
      center[0] = 0.5;
      for (unsigned int d = 1; d < dim; ++d)
        center[d] = 0.004;

      dealii::Tensor<1, dim, number> normal;
      normal[0] = 1.;

      const auto level_set =
        std::make_shared<dealii::Functions::SignedDistance::Plane<dim>>(center, normal);
      this->attach_field_function(level_set, "level_set", "compressible_flow");

      // set inflow function at unfitted object
      this->attach_field_function(initial_condition, "unfitted_inflow", "compressible_flow");
    }

    /**
     * @brief Performs post-processing by evaluating and outputting error norms.
     *
     * @param generic_data_out A generic utility for managing simulation output data.
     */
    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      SteadyInflowField<dim, number> reference_values;
      this->print_relative_norm(generic_data_out, reference_values, "norm");
    }

  private:
    /// Enumeration for the boundary id's.
    enum BoundaryID
    {
      subsonic_outflow_with_fixed_energy,
      slip_wall
    };

    /**
     * @brief Set boundary id's for fitted boundaries.
     *
     * @param triangulation Triangulation object.
     */
    void
    set_fitted_boundary_id(const auto &triangulation) const
    {
      for (const auto &cell : triangulation.cell_iterators())
        if (cell->at_boundary())
          {
            for (const auto &face : cell->face_iterators())
              {
                const auto center = face->center();
                // outflow boundary with fixed energy
                if ((std::fabs(center(0)) > 1. - 1e-12))
                  face->set_boundary_id(BoundaryID::subsonic_outflow_with_fixed_energy);
                // slip wall boundary conditions
                else if ((std::fabs(center(1)) < 1e-12) or (std::fabs(center(1)) > 0.008 - 1e-12))
                  face->set_boundary_id(BoundaryID::slip_wall);
              }
          }
    }
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
