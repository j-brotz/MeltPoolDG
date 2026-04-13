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
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/functions.hpp>

#include <meltpooldg/utilities/cpp23_functions.h>

#include <memory>

#include "../compressible_flow_case.hpp"

namespace MeltPoolDG::Simulation::CompressibleFlow
{
  /**
   * A function class that computes the conservative variable values for the compressible
   * Navier–Stokes equations based on provided functions for the primitive variables:
   * density, velocity, and specific total energy.
   *
   * Optionally, a time-dependent ramp-up can be applied to the velocity.
   */
  template <int dim, typename number>
  class MultiComponentConservativeFunction : public dealii::Function<dim, number>
  {
  public:
    /**
     * Constructor, stores the passed arguments internally.
     */
    MultiComponentConservativeFunction(
      const number                                         initial_time,
      const std::shared_ptr<dealii::Function<dim, number>> density,
      const std::shared_ptr<dealii::Function<dim, number>> velocity,
      const std::shared_ptr<dealii::Function<dim, number>> energy,
      const std::shared_ptr<dealii::Function<dim, number>> species)
      : dealii::Function<dim, number>(dim + 2 + species->n_components, initial_time)
      , density(density)
      , velocity(velocity)
      , energy(energy)
      , species(species)
    {}

    /**
     * Set the time for the function.
     *
     * @param new_time Time to be set.
     */
    void
    set_time(const number new_time) override
    {
      dealii::Function<dim>::set_time(new_time);
      density->set_time(new_time);
      velocity->set_time(new_time);
      energy->set_time(new_time);
      species->set_time(new_time);
    }

    /**
     * @brief Evaluates the conservative variable value for a given component at a specified location.
     *
     * This function returns the value of the conservative variable corresponding to the specified
     * component at the given spatial location @p loc. The value is computed using the density,
     * velocity, and energy functions provided to the object during construction.
     *
     * If a ramp-up duration is defined, the velocity components are scaled according to the
     * selected ramp-up function depending on the current time.
     *
     * If no inflow ramp-up duration is specified (i.e., @p inflow_ramp_up_duration = 0),
     * the returned value corresponds directly to the fully developed flow field.
     *
     * @param loc Coordinates at which the function value is evaluated.
     * @param component  The index of the conservative variable component to evaluate.
     *
     * @return The value of the specified conservative variable component at @p loc.
     */
    number
    value(const dealii::Point<dim, number> &loc, const unsigned int component) const override
    {
      const auto compute_conserved_energy = [&](const dealii::Point<dim, number> &loc) -> number {
        dealii::Vector<number> velocity_vec(dim);
        velocity->vector_value(loc, velocity_vec);
        return density->value(loc, 0) * (energy->value(loc, 0) + 0.5 * velocity_vec.norm_sqr());
      };

      if (component == 0)
        return density->value(loc, 0);
      else if (component > 0 and component <= dim)
        return density->value(loc, 0) * velocity->value(loc, component - 1);
      else if (component == dim + 1)
        return compute_conserved_energy(loc);
      else if (component < this->n_components)
        return density->value(loc, 0) * species->value(loc, component - dim - 2);
      else
        AssertThrow(false,
                    dealii::ExcMessage("Invalid component for conservative variable 2 value: " +
                                       std::to_string(component) + ". Only components from 0 to " +
                                       std::to_string(dim + 1 + species->n_components) +
                                       " are valid."));
    }

  private:
    /// Function describing the density field.
    const std::shared_ptr<dealii::Function<dim, number>> density;

    /// Vectorial function describing the velocity field.
    const std::shared_ptr<dealii::Function<dim, number>> velocity;

    /// Function describing the field of the specific total energy.
    const std::shared_ptr<dealii::Function<dim, number>> energy;

    /// Function describing the field of the species mass fraction. This is only used for
    /// multi-component flow cases and is ignored for single-species flow cases.
    const std::shared_ptr<dealii::Function<dim, number>> species;
  };


  /**
   * Struct for storing boundary condition information for a specific boundary and creating
   * a corresponding function object for use in simulations.
   */
  template <int dim, typename number>
  struct MultiComponentBoundaryCondition
  {
    /// Type of the boundary condition (default: slip wall)
    MeltPoolDG::CompressibleFlow::BoundaryConditionType type =
      MeltPoolDG::CompressibleFlow::BoundaryConditionType::slip_wall;

    /// Prescribed density at the boundary (used if required by the boundary condition type)
    std::string density;

    /// Prescribed velocity vector at the boundary (used if required by the boundary condition type)
    std::string velocity;

    /// Prescribed pressure at the boundary (used if required by the boundary condition type)
    std::string pressure;

    /// Prescribed specific total energy at the boundary (used if required by the boundary condition
    /// type). For a subsonic outflow with fixed energy, this corresponds to the Dirichlet value for
    /// the conservative variable ρ*E, not just the specific total energy.
    std::string energy;

    std::string species_mass_fractions;

    /**
     * Creates a function object representing the boundary condition based on the data stored
     * in this struct. For example, for a fixed-pressure outflow, this returns a
     * dealii::Function representing the pressure field at the boundary.
     *
     * @param start_time Initial time used to initialize the function.
     *
     * @return Shared pointer to a dealii::Function representing the boundary condition.
     */
    std::shared_ptr<dealii::Function<dim>>
    create_boundary_function(const number start_time, const int n_species) const
    {
      AssertThrow(
        n_species > 1,
        dealii::ExcMessage(
          "The multi component flow case can currently not handle single component flow. Please specify at least two species in the input file."));
      switch (type)
        {
            case (MeltPoolDG::CompressibleFlow::BoundaryConditionType::inflow): {
              auto density_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
              density_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                density,
                typename dealii::FunctionParser<dim>::ConstMap());

              auto velocity_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(dim, start_time);
              velocity_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                velocity,
                typename dealii::FunctionParser<dim>::ConstMap());

              auto energy_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
              energy_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                energy,
                typename dealii::FunctionParser<dim>::ConstMap());

              auto species_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(n_species - 1, start_time);
              species_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                species_mass_fractions,
                typename dealii::FunctionParser<dim>::ConstMap());

              return std::make_shared<MultiComponentConservativeFunction<dim, number>>(
                start_time,
                density_boundary_function,
                velocity_boundary_function,
                energy_boundary_function,
                species_boundary_function);
            }
            case (
              MeltPoolDG::CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_energy): {
              auto energy_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
              energy_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                energy,
                typename dealii::FunctionParser<dim>::ConstMap());
              return energy_boundary_function;
            }
            case (MeltPoolDG::CompressibleFlow::BoundaryConditionType::
                    subsonic_outflow_fixed_pressure): {
              auto pressure_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
              pressure_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                pressure,
                typename dealii::FunctionParser<dim>::ConstMap());
              return pressure_boundary_function;
            }
          default:
            return std::make_shared<dealii::Functions::ConstantFunction<dim>>(0.);
        }
    }
  };


  template <int dim, typename number>
  class SimulationMultiComponentGenericHyperRectangle final
    : public MeltPoolDG::CompressibleFlow::Case<dim, number>
  {
  public:
    SimulationMultiComponentGenericHyperRectangle(std::string    parameter_file,
                                                  const MPI_Comm mpi_communicator)
      : MeltPoolDG::CompressibleFlow::Case<dim, number>(parameter_file, mpi_communicator)
    {}

    void
    create_spatial_discretization() override
    {
      auto create_point_from_container =
        []<typename T>(const T &container) -> dealii::Point<dim, number> {
        dealii::Point<dim, number> point;
        if constexpr (dim == 1)
          point = dealii::Point<dim, number>(container[0]);
        else if constexpr (dim == 2)
          point = dealii::Point<dim, number>(container[0], container[1]);
        else if constexpr (dim == 3)
          point = dealii::Point<dim, number>(container[0], container[1], container[2]);
        return point;
      };

      if constexpr (dim == 1)
        this->triangulation =
          std::make_shared<dealii::parallel::shared::Triangulation<dim>>(this->mpi_communicator);
      else
        this->triangulation = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
          this->mpi_communicator);

      dealii::Point<dim, number> dimensions =
        create_point_from_container(scenario_data.domain_dimensions);

      dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        scenario_data.domain_base_discretization,
                                                        dealii::Point<dim, number>(),
                                                        dimensions,
                                                        true);

      AssertThrow(scenario_data.periodic_boundaries.size() == dim,
                  dealii::ExcMessage("Invalid size for periodic boundaries vector. "
                                     "Expected size " +
                                     std::to_string(dim) + ", but got size " +
                                     std::to_string(scenario_data.periodic_boundaries.size()) +
                                     "."));

      for (unsigned int d = 0; d < dim; ++d)
        {
          if (scenario_data.periodic_boundaries[d])
            {
              std::vector<dealii::GridTools::PeriodicFacePair<
                typename dealii::Triangulation<dim>::cell_iterator>>
                periodicity_vector;

              dealii::GridTools::collect_periodic_faces(
                *(this->triangulation), 2 * d, 2 * d + 1, d, periodicity_vector);
              this->triangulation->add_periodicity(periodicity_vector);
            }
        }

      this->triangulation->refine_global(this->parameters.base.global_refinements);

      // Perform additional mesh refinement in the user-defined region of interest
      if (scenario_data.region_of_interest_refinement_times > 0)
        {
          AssertThrow(scenario_data.region_of_interest_corner[0].size() == dim &&
                        scenario_data.region_of_interest_corner[1].size() == dim,
                      dealii::ExcMessage(
                        "Invalid size for region-of-interest corner points. "
                        "Both points must have size " +
                        std::to_string(dim) + ", but got sizes " +
                        std::to_string(scenario_data.region_of_interest_corner[0].size()) +
                        " and " +
                        std::to_string(scenario_data.region_of_interest_corner[1].size()) + "."));


          std::vector<AMR::AMRRegion<dim, number>> regions;
          regions.emplace_back(dealii::BoundingBox<dim, number>(
            {create_point_from_container(scenario_data.region_of_interest_corner[0]),
             create_point_from_container(scenario_data.region_of_interest_corner[1])}));
          for (unsigned i = 0; i < scenario_data.region_of_interest_refinement_times; ++i)
            {
              AMR::set_refinement_flags_in_regions<dim, number>(*this->triangulation, regions);
              this->triangulation->execute_coarsening_and_refinement();
            }
        }
    }

    void
    set_boundary_conditions() override
    {
      for (unsigned i = 0; i < scenario_data.boundary_conditions.size(); ++i)
        this->attach_boundary_condition(
          std::make_pair(i,
                         scenario_data.boundary_conditions[i].create_boundary_function(
                           this->parameters.time_stepping.start_time,
                           this->parameters.material.number_of_species)),
          boundary_type_to_string_map.at(scenario_data.boundary_conditions[i].type),
          "compressible_flow");
    }

    void
    set_field_conditions() override
    {
      auto density = std::make_shared<dealii::FunctionParser<dim>>(1);
      density->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                          scenario_data.initial_condition.density,
                          typename dealii::FunctionParser<dim>::ConstMap());
      auto velocity = std::make_shared<dealii::FunctionParser<dim>>(dim);
      velocity->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                           scenario_data.initial_condition.velocity,
                           typename dealii::FunctionParser<dim>::ConstMap());
      auto energy = std::make_shared<dealii::FunctionParser<dim>>(1);
      energy->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                         scenario_data.initial_condition.energy,
                         typename dealii::FunctionParser<dim>::ConstMap());

      auto mass_fractions = std::make_shared<dealii::FunctionParser<dim>>(
        this->parameters.material.number_of_species - 1);
      mass_fractions->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                                 scenario_data.initial_condition.species_mass_fractions,
                                 typename dealii::FunctionParser<dim>::ConstMap());

      auto initial_condition = std::make_shared<MultiComponentConservativeFunction<dim, number>>(
        this->parameters.time_stepping.start_time, density, velocity, energy, mass_fractions);

      this->attach_initial_condition(initial_condition, "compressible_flow");
    }

    void
    do_postprocessing(const GenericDataOut<dim, number> &generic_data_out) const override
    {
      auto density = std::make_shared<dealii::FunctionParser<dim>>(1);
      density->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                          scenario_data.initial_condition.density,
                          typename dealii::FunctionParser<dim>::ConstMap());
      auto velocity = std::make_shared<dealii::FunctionParser<dim>>(dim);
      velocity->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                           scenario_data.initial_condition.velocity,
                           typename dealii::FunctionParser<dim>::ConstMap());
      auto energy = std::make_shared<dealii::FunctionParser<dim>>(1);
      energy->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                         scenario_data.initial_condition.energy,
                         typename dealii::FunctionParser<dim>::ConstMap());
      auto mass_fractions = std::make_shared<dealii::FunctionParser<dim>>(
        this->parameters.material.number_of_species - 1);
      mass_fractions->initialize(dealii::FunctionParser<dim>::default_variable_names(),
                                 scenario_data.initial_condition.species_mass_fractions,
                                 typename dealii::FunctionParser<dim>::ConstMap());

      MultiComponentConservativeFunction<dim, number> reference_values(
        generic_data_out.get_time(), density, velocity, energy, mass_fractions);

      using DataToPrint =
        typename MeltPoolDG::CompressibleFlow::Case<dim, number>::DataPostprocessorData;

      std::vector<DataToPrint> postprocessor_data_vector;
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "density", .reference_function = *density});
      const auto momentum =
        Functions::ExtractedComponentsFunction<dim, number>(reference_values, 1, dim);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "momentum", .reference_function = momentum});
      const auto total_energy =
        Functions::ExtractedComponentsFunction<dim, number>(reference_values, 1 + dim, 1);
      postprocessor_data_vector.emplace_back(
        DataToPrint{.name = "total energy", .reference_function = total_energy});

      std::vector<Functions::ExtractedComponentsFunction<dim, number>> partial_densities;
      for (unsigned int species = 0; species < this->parameters.material.number_of_species - 1;
           ++species)
        {
          partial_densities.emplace_back(Functions::ExtractedComponentsFunction<dim, number>(
            reference_values, 2 + dim + species, 1));
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
        if constexpr (dim >= 1)
          {
            add_user_boundary_condition_parameters(prm,
                                                   "boundary x min",
                                                   scenario_data.boundary_conditions[0]);
            add_user_boundary_condition_parameters(prm,
                                                   "boundary x max",
                                                   scenario_data.boundary_conditions[1]);
          }
        if constexpr (dim >= 2)
          {
            add_user_boundary_condition_parameters(prm,
                                                   "boundary y min",
                                                   scenario_data.boundary_conditions[2]);
            add_user_boundary_condition_parameters(prm,
                                                   "boundary y max",
                                                   scenario_data.boundary_conditions[3]);
          }
        if constexpr (dim == 3)
          {
            add_user_boundary_condition_parameters(prm,
                                                   "boundary z min",
                                                   scenario_data.boundary_conditions[4]);
            add_user_boundary_condition_parameters(prm,
                                                   "boundary z max",
                                                   scenario_data.boundary_conditions[5]);
          }

        prm.enter_subsection("domain");
        {
          prm.add_parameter(
            "size",
            scenario_data.domain_dimensions,
            "Physical dimensions of the computational domain. "
            "Specify comma-separated values: the first for the x-direction, the second for the y-direction, "
            "and optionally the third for the z-direction.");
          prm.add_parameter(
            "base grid resolution",
            scenario_data.domain_base_discretization,
            "Number of base cells in each spatial direction. "
            "Provide comma-separated values: the first for the x-direction, the second for the y-direction, "
            "and optionally the third for the z-direction. "
            "Note: the final mesh resolution may differ depending on other parameters "
            "(e.g., global refinements).");
          prm.add_parameter(
            "periodic boundary conditions",
            scenario_data.periodic_boundaries,
            "Specify whether the boundaries in each direction are periodic. "
            "Provide comma-separated boolean values (true/false) for each direction, e.g., 'true; false; false' for a 3D simulation with periodicity only in the x-direction.");

          prm.enter_subsection("region of interest");
          {
            prm.add_parameter(
              "first corner",
              scenario_data.region_of_interest_corner[0],
              "Coordinates of the bottom corner of the region of interest. "
              "Provide comma-separated values: the first for the x-coordinate, the second for the y-coordinate, "
              "and optionally the third for the z-coordinate.");
            prm.add_parameter(
              "second corner",
              scenario_data.region_of_interest_corner[1],
              "Coordinates of the top corner of the region of interest. "
              "Provide comma-separated values: the first for the x-coordinate, the second for the y-coordinate, "
              "and optionally the third for the z-coordinate.");
            prm.add_parameter(
              "refinements",
              scenario_data.region_of_interest_refinement_times,
              "Number of additional refinement levels to apply inside the defined region. "
              "A value of zero disables region-specific refinement.");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
        prm.enter_subsection("initial conditions");
        {
          prm.add_parameter("density",
                            scenario_data.initial_condition.density,
                            "Initial density field value or spatial distribution.");
          prm.add_parameter(
            "velocity",
            scenario_data.initial_condition.velocity,
            "Initial velocity field. "
            "Specify components separated by semicolons, e.g., '1.0; 0.0; 0.0' for a 3D simulation.");
          prm.add_parameter("energy",
                            scenario_data.initial_condition.energy,
                            "Initial specific energy field value or distribution.");
          prm.add_parameter(
            "species mass fractions",
            scenario_data.initial_condition.species_mass_fractions,
            "Initial species mass fractions field value or distribution (only used for multi-component flow cases).");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      return this->parameters.base.do_print_parameters;
    }

  private:
    /**
     * This function reads boundary condition data for a single boundary from the user input file.
     * To do this, one provides the name of the subsection in the input file that describes the
     * boundary condition and a BoundaryCondition object, in which the provided data is stored.
     *
     * @param prm Parameter handler used to read the input file.
     * @param boundary_location Name of the subsection containing the boundary data.
     * @param boundary_condition BoundaryCondition object where the read data will be stored.
     */
    void
    add_user_boundary_condition_parameters(
      dealii::ParameterHandler                     &prm,
      const std::string                            &boundary_location,
      MultiComponentBoundaryCondition<dim, number> &boundary_condition)
    {
      prm.enter_subsection(boundary_location);
      {
        prm.add_parameter("type", boundary_condition.type, "Type of the boundary.");
        prm.add_parameter(
          "density",
          boundary_condition.density,
          "Prescribed density at the boundary (depending on type of boundary condition).");
        prm.add_parameter(
          "velocity",
          boundary_condition.velocity,
          "Prescribed velocity at the boundary (depending on type of boundary condition). "
          "Velocity components are separated by semicolon, e.g., '1.0; 0.0; 0.0' for 3D.");
        prm.add_parameter(
          "pressure",
          boundary_condition.pressure,
          "Prescribed pressure at the boundary (depending on type of boundary condition).");
        prm.add_parameter(
          "energy",
          boundary_condition.energy,
          "Prescribed energy at the boundary (depending on type of boundary condition).");
        prm.add_parameter(
          "species mass fractions",
          boundary_condition.species_mass_fractions,
          "Prescribed mass fractions of species at the boundary (only used for multi-component flow cases).");
      }
      prm.leave_subsection();
    }

    /// Number of domain boundaries
    constexpr static unsigned n_boundaries = 2 * dim;

    struct
    {
      /// Array of boundary condition objects describing the type and values of the boundaries.
      /// The array index corresponds to the boundary ID to which the boundary condition applies.
      std::array<MultiComponentBoundaryCondition<dim, number>, n_boundaries> boundary_conditions;

      struct
      {
        /// Function describing the initial density field
        std::string density;
        /// Function describing the initial velocity field
        std::string velocity;
        /// Function describing the initial specific total energy
        std::string energy;
        /// Function describing the initial species mass fractions (only used for multi-component
        /// flow cases)
        std::string species_mass_fractions;
      } initial_condition;

      /// Sizes of the domain in each dimension. The x, y, and z sizes are given at indices 0, 1,
      /// and 2, respectively.
      std::vector<number> domain_dimensions;

      /// Discretization of the domain in each dimension. The x, y, and z discretizations are given
      /// at indices 0, 1, and 2, respectively.
      std::vector<unsigned> domain_base_discretization;

      /// Defines the opposite corners of an axis-aligned box in physical space. The region enclosed
      /// by these points may be subject to additional refinement, depending on the value of
      /// @ref region_of_interest_refinement_times.
      std::array<std::vector<number>, 2> region_of_interest_corner;

      /// If greater than zero, cells whose centers lie within the region defined by
      /// @ref region_of_interest_corner will be refined this many additional times
      /// beyond the global refinement level.
      unsigned region_of_interest_refinement_times = 0;

      std::vector<bool> periodic_boundaries = std::vector<bool>(dim, false);
    } scenario_data;

    /// Mapping to translate between the enum used to specify boundary conditions in the input file
    /// and the corresponding string names used in the simulation case.
    const std::map<MeltPoolDG::CompressibleFlow::BoundaryConditionType, std::string>
      boundary_type_to_string_map = {
        {MeltPoolDG::CompressibleFlow::BoundaryConditionType::inflow, "inflow"},
        {MeltPoolDG::CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_pressure,
         "outflow_fixed_pressure"},
        {MeltPoolDG::CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_energy,
         "outflow_fixed_energy"},
        {MeltPoolDG::CompressibleFlow::BoundaryConditionType::slip_wall, "slip_wall"},
        {MeltPoolDG::CompressibleFlow::BoundaryConditionType::no_slip_wall, "no_slip_wall"},
      };
  };
} // namespace MeltPoolDG::Simulation::CompressibleFlow
