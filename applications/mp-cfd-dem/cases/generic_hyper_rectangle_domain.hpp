#pragma once

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/utilities/amr_regions.hpp>
#include <meltpooldg/utilities/better_enum.hpp>

#include <memory>
#include <numbers>

#include "../cfd_dem_case.hpp"

namespace MeltPoolDG::Simulation::CfdDem
{

  /// Enum for the type of ramp up function used for the velocity at an inflow boundary.
  BETTER_ENUM(RampUpType, char, none, linear, exponential, cosine);

  /**
   * A function class that computes the conservative variable values for the compressible
   * Navier–Stokes equations based on provided functions for the primitive variables:
   * density, velocity, and specific total energy.
   *
   * Optionally, a time-dependent ramp-up can be applied to the velocity.
   */
  template <int dim, typename number>
  class ConservativeFunction : public dealii::Function<dim, number>
  {
  public:
    /**
     * Constructor, stores the passed arguments internally.
     */
    ConservativeFunction(const number                                         initial_time,
                         const std::shared_ptr<dealii::Function<dim, number>> density,
                         const std::shared_ptr<dealii::Function<dim, number>> velocity,
                         const std::shared_ptr<dealii::Function<dim, number>> energy,
                         const number     inflow_ramp_up_duration = 0.,
                         const RampUpType ramp_up_type            = RampUpType::linear)
      : dealii::Function<dim, number>(dim + 2, initial_time)
      , density(density)
      , velocity(velocity)
      , energy(energy)
      , inflow_ramp_up_duration(inflow_ramp_up_duration)
      , ramp_up_type(ramp_up_type)
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
      const auto inflow_ramp_up_scaling = [&]() -> number {
        if (this->get_time() > inflow_ramp_up_duration or inflow_ramp_up_duration == 0. or
            ramp_up_type == RampUpType::none)
          return 1.;

        switch (ramp_up_type)
          {
            case RampUpType::linear:
              return this->get_time() / inflow_ramp_up_duration;
              break;
            case RampUpType::exponential:
              return (std::exp(this->get_time() / inflow_ramp_up_duration) - 1) /
                     (std::numbers::e - 1);
              break;
            case RampUpType::cosine:
              return 0.5 *
                     (1 - std::cos(std::numbers::pi * this->get_time() / inflow_ramp_up_duration));
              break;
            default:
              AssertThrow(false, dealii::ExcInternalError());
          }
      };

      const auto compute_conserved_energy = [&](const dealii::Point<dim, number> &loc) -> number {
        dealii::Vector<number> velocity_vec(dim);
        velocity->vector_value(loc, velocity_vec);
        velocity_vec *= inflow_ramp_up_scaling() * density->value(loc, 0);
        return density->value(loc, 0) * (energy->value(loc, 0) + 0.5 * velocity_vec.norm_sqr());
      };

      switch (component)
        {
          case 0:
            return density->value(loc, 0);
            break;
          case 1:
            return density->value(loc, 0) * inflow_ramp_up_scaling() * velocity->value(loc, 0);
            break;
          case 2:
            return density->value(loc, 0) * inflow_ramp_up_scaling() * velocity->value(loc, 1);
            break;
          case 3:
            if constexpr (dim == 3)
              return density->value(loc, 0) * inflow_ramp_up_scaling() * velocity->value(loc, 2);
            else
              return compute_conserved_energy(loc);
            break;
          case 4:
            if constexpr (dim == 3)
              return compute_conserved_energy(loc);
            else
              AssertThrow(false, dealii::ExcInternalError());
            break;
          default:
            AssertThrow(false, dealii::ExcInternalError());
        }
    }

  private:
    /// Function describing the density field.
    const std::shared_ptr<dealii::Function<dim, number>> density;

    /// Vectorial function describing the velocity field.
    const std::shared_ptr<dealii::Function<dim, number>> velocity;

    /// Function describing the field of the specific total energy.
    const std::shared_ptr<dealii::Function<dim, number>> energy;

    /// In the case of an inflow boundary the time for which the inflow velocity is ramped-up.
    const number inflow_ramp_up_duration;

    /// In the case of an inflow boundary the type of ramp-up function for the velocity.
    const RampUpType ramp_up_type;
  };


  /**
   * Struct for storing boundary condition information for a specific boundary and creating
   * a corresponding function object for use in simulations.
   */
  template <int dim, typename number>
  struct BoundaryCondition
  {
    /// Type of the boundary condition (default: slip wall)
    CompressibleFlow::BoundaryConditionType type =
      CompressibleFlow::BoundaryConditionType::slip_wall;

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

    /// Duration over which the inflow velocity is ramped up (only relevant for inflow boundaries)
    number inflow_ramp_up_duration = 0.;

    /// Type of ramp-up function applied to the inflow velocity (only relevant for inflow
    /// boundaries)
    RampUpType inflow_ramp_up_type = RampUpType::linear;

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
    create_boundary_function(const number start_time)
    {
      switch (type)
        {
            case (CompressibleFlow::BoundaryConditionType::inflow): {
              auto density_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
              density_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                density,
                typename dealii::FunctionParser<dim>::ConstMap());

              auto velocity_boundary_function = std::make_shared<dealii::FunctionParser<dim>>(dim);
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

              return std::make_shared<ConservativeFunction<dim, number>>(start_time,
                                                                         density_boundary_function,
                                                                         velocity_boundary_function,
                                                                         energy_boundary_function,
                                                                         inflow_ramp_up_duration,
                                                                         inflow_ramp_up_type);
            }
            case (CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_energy): {
              auto energy_boundary_function =
                std::make_shared<dealii::FunctionParser<dim>>(1, start_time);
              energy_boundary_function->initialize(
                dealii::FunctionParser<dim>::default_variable_names(),
                energy,
                typename dealii::FunctionParser<dim>::ConstMap());
              return energy_boundary_function;
            }
            case (CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_pressure): {
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
      auto create_point_from_container =
        []<typename T>(const T &container) -> dealii::Point<dim, number> {
        dealii::Point<dim, number> point;
        if constexpr (dim == 2)
          point = dealii::Point<dim, number>(container[0], container[1]);
        else if constexpr (dim == 3)
          point = dealii::Point<dim, number>(container[0], container[1], container[2]);
        return point;
      };

      this->triangulation =
        std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

      dealii::Point<dim, number> dimensions =
        create_point_from_container(scenario_data.domain_dimensions);

      dealii::GridGenerator::subdivided_hyper_rectangle(*this->triangulation,
                                                        scenario_data.domain_base_discretization,
                                                        dealii::Point<dim, number>(),
                                                        dimensions,
                                                        true);

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
                           this->parameters.time_stepping.start_time)),
          boundary_type_to_string_map.at(scenario_data.boundary_conditions[i].type),
          "cfd_dem");
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

      auto initial_condition = std::make_shared<ConservativeFunction<dim, number>>(
        this->parameters.time_stepping.start_time, density, velocity, energy);

      this->attach_initial_condition(initial_condition, "cfd_dem");
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

      ConservativeFunction<dim, number> reference_values(generic_data_out.get_time(),
                                                         density,
                                                         velocity,
                                                         energy);
      this->print_relative_norm(generic_data_out, reference_values, "norm");
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
        add_user_boundary_condition_parameters(prm,
                                               "boundary x min",
                                               scenario_data.boundary_conditions[0]);
        add_user_boundary_condition_parameters(prm,
                                               "boundary x max",
                                               scenario_data.boundary_conditions[1]);
        add_user_boundary_condition_parameters(prm,
                                               "boundary y min",
                                               scenario_data.boundary_conditions[2]);
        add_user_boundary_condition_parameters(prm,
                                               "boundary y max",
                                               scenario_data.boundary_conditions[3]);
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
    add_user_boundary_condition_parameters(dealii::ParameterHandler       &prm,
                                           const std::string              &boundary_location,
                                           BoundaryCondition<dim, number> &boundary_condition)
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
          "inflow ramp-up duration",
          boundary_condition.inflow_ramp_up_duration,
          "In the case of an inflow boundary, the time ramp-up duration of the inflow velocity.");
        prm.add_parameter(
          "inflow ramp-up type",
          boundary_condition.inflow_ramp_up_type,
          "In the case of an inflow boundary, the ramp-up type of the inflow velocity. "
          "Supported options are 'linear', 'exponential' and 'cosine'.");
      }
      prm.leave_subsection();
    }

    struct
    {
      /// Array of boundary condition objects describing the type and values of the boundaries.
      /// The array index corresponds to the boundary ID to which the boundary condition applies.
      std::array<BoundaryCondition<dim, number>, 2 * dim> boundary_conditions;

      struct
      {
        /// Function describing the initial density field
        std::string density;
        /// Function describing the initial velocity field
        std::string velocity;
        /// Function describing the initial specific total energy
        std::string energy;
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

    } scenario_data;

    /// Mapping to translate between the enum used to specify boundary conditions in the input file
    /// and the corresponding string names used in the simulation case.
    const std::map<CompressibleFlow::BoundaryConditionType, std::string>
      boundary_type_to_string_map = {
        {CompressibleFlow::BoundaryConditionType::inflow, "inflow"},
        {CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_pressure,
         "outflow_fixed_pressure"},
        {CompressibleFlow::BoundaryConditionType::subsonic_outflow_fixed_energy,
         "outflow_fixed_energy"},
        {CompressibleFlow::BoundaryConditionType::slip_wall, "slip_wall"},
        {CompressibleFlow::BoundaryConditionType::no_slip_wall, "no_slip_wall"},
      };
  };
} // namespace MeltPoolDG::Simulation::CfdDem
