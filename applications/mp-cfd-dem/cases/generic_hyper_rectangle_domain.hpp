#pragma once

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/case_utils.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/better_enum.hpp>

#include <memory>

#include "../cfd_dem_case.hpp"

namespace MeltPoolDG::Simulation::CfdDem
{
  template <int dim, typename number>
  class SimulationGenericHyperRectangleDomain final : public CfdDemCase<dim, number>
  {
  public:
    SimulationGenericHyperRectangleDomain(std::string    parameter_file,
                                          const MPI_Comm mpi_communicator)
      : CfdDemCase<dim, number>(parameter_file, mpi_communicator)
    {
      AssertThrow(
        dim > 1,
        dealii::ExcMessage(
          "The particles in channel flow case requires dim > 1 but the dimension is set to dim = " +
          std::to_string(dim) + "."));
    }

    void
    create_spatial_discretization() override
    {
      triangulation_creator.create_triangulation(this->triangulation,
                                                 this->mpi_communicator,
                                                 this->parameters.base.global_refinements);
    }

    void
    set_boundary_conditions() override
    {
      for (unsigned i = 0; i < boundary_conditions.size(); ++i)
        this->attach_boundary_condition(
          std::make_pair(i,
                         boundary_conditions[i].create_boundary_function(
                           this->parameters.time_stepping.start_time,
                           this->parameters.material.number_of_species)),
          CompressibleFlow::BoundaryConditions<dim, number>::boundary_type_to_string_map.at(
            boundary_conditions[i].type),
          "cfd_dem");
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(initial_condition.create_initial_condition_function(
                                       this->parameters.time_stepping.start_time,
                                       this->parameters.material.number_of_species),
                                     "cfd_dem");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      std::shared_ptr<dealii::Function<dim, number>> reference_values =
        initial_condition.create_initial_condition_function(
          generic_data_out.get_time(), this->parameters.material.number_of_species);

      this->print_relative_norm(generic_data_out, *reference_values, "norm");
    }

    /**
     * Reads boundary conditions, initial conditions, and domain size parameters from
     * the user input file.
     */
    bool
    add_simulation_specific_parameters(dealii::ParameterHandler &prm) override
    {
      prm.enter_subsection("case setup");
      {
        MeltPoolDG::CompressibleFlow::add_hyper_rectangle_custom_boundary_condition_parameters(
          prm, boundary_conditions);

        triangulation_creator.add_parameters(prm);

        initial_condition.add_parameters(prm);
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /// Number of domain boundaries
    constexpr static unsigned n_boundaries = 2 * dim;
    /// Array of boundary condition objects describing the type and values of the boundaries.
    /// The array index corresponds to the boundary ID to which the boundary condition applies.
    std::array<MeltPoolDG::CompressibleFlow::InputDefinedBoundaryCondition<dim, number>,
               n_boundaries>
      boundary_conditions;

    MeltPoolDG::CompressibleFlow::InputDefinedInitialCondition<dim, number> initial_condition;

    MeltPoolDG::CompressibleFlow::InputDefinedSubdividedHyperRectangleDomain<dim, number>
      triangulation_creator;
  };
} // namespace MeltPoolDG::Simulation::CfdDem
