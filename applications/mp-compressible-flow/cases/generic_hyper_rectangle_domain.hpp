#pragma once

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/case_utils.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/functions.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <memory>

#include "../compressible_flow_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  template <int dim, typename number>
  class SimulationGenericHyperRectangleDomain final
    : public MeltPoolDG::CompressibleFlow::Case<dim, number>
  {
  public:
    SimulationGenericHyperRectangleDomain(std::string    parameter_file,
                                          const MPI_Comm mpi_communicator)
      : MeltPoolDG::CompressibleFlow::Case<dim, number>(parameter_file, mpi_communicator)
    {}

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
          MeltPoolDG::CompressibleFlow::BoundaryConditions<dim, number>::boundary_type_to_string_map
            .at(boundary_conditions[i].type),
          "compressible_flow");
    }

    void
    set_field_conditions() override
    {
      this->attach_initial_condition(initial_condition.create_initial_condition_function(
                                       this->parameters.time_stepping.start_time,
                                       this->parameters.material.number_of_species),
                                     "compressible_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      std::shared_ptr<dealii::Function<dim, number>> reference_values =
        initial_condition.create_initial_condition_function(
          generic_data_out.get_time(), this->parameters.material.number_of_species);

      using DataToPrint =
        typename MeltPoolDG::CompressibleFlow::Case<dim, number>::DataPostprocessorData;

      std::vector<DataToPrint> postprocessor_data_vector;
      const auto               density =
        Functions::ExtractedComponentsFunction<dim, number>(*reference_values, 0, 1);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "density", .reference_function = density});
      const auto momentum =
        Functions::ExtractedComponentsFunction<dim, number>(*reference_values, 1, dim);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "momentum", .reference_function = momentum});
      const auto total_energy =
        Functions::ExtractedComponentsFunction<dim, number>(*reference_values, 1 + dim, 1);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "total energy", .reference_function = total_energy});

      std::vector<Functions::ExtractedComponentsFunction<dim, number>> partial_densities;
      for (unsigned int species = 0; species < this->parameters.material.number_of_species - 1;
           ++species)
        {
          partial_densities.emplace_back(Functions::ExtractedComponentsFunction<dim, number>(
            *reference_values, 2 + dim + species, 1));
          postprocessor_data_vector.emplace_back(
            DataToPrint{.name =
                          this->parameters.material.species_data[species].name + " partial density",
                        .reference_function = partial_densities[species]});
        }

      this->print_relative_norm_fitted(generic_data_out, postprocessor_data_vector, "norm");
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
} // namespace MeltPoolDG::Simulation::CompressibleFlow
