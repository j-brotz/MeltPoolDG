/**
 * @brief This file contains various functions that can be used to set and evaluate boundary conditions for the compressible flow solver. The functions can be directly used with the `BoundaryConditions` class, which provides an interface to manage and evaluate the different boundary conditions in the solver.
 */

#pragma once

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/function.h>

#include <meltpooldg/compressible_flow/boundary_conditions.hpp>
#include <meltpooldg/compressible_flow/data_types.hpp>
#include <meltpooldg/utilities/better_enum.hpp>

#include <memory>

namespace MeltPoolDG::CompressibleFlow
{
  /// Enum for the type of ramp up function used for the velocity at an inflow boundary.
  BETTER_ENUM(RampUpType, char, none, linear, exponential, cosine);

  /**
   * A function class that computes the conservative variable values for the
   * compressible Navier–Stokes equations based on provided functions for the primitive variables:
   * density, velocity, specific inner energy and species mass fractions.
   *
   * Optionally, a time-dependent ramp-up can be applied to the velocity.
   */
  template <int dim, typename number>
  class ConservativeVariablesFunction : public dealii::Function<dim, number>
  {
  public:
    /**
     * Constructor, stores the passed arguments internally.
     *
     * @param initial_time Initial time used to initialize the function.
     * @param density Function representing the density field.
     * @param velocity Function representing the velocity field.
     * @param inner_energy Function representing the specific inner energy field.
     * @param mass_fractions Function representing the species mass fractions.
     */
    ConservativeVariablesFunction(
      const number                                         initial_time,
      const std::shared_ptr<dealii::Function<dim, number>> density,
      const std::shared_ptr<dealii::Function<dim, number>> velocity,
      const std::shared_ptr<dealii::Function<dim, number>> inner_energy,
      const std::shared_ptr<dealii::Function<dim, number>> mass_fractions);

    /**
     * As above, stores the passed arguments internally but for single-species flow cases.
     *
     * @param initial_time Initial time used to initialize the function.
     * @param density Function representing the density field.
     * @param velocity Function representing the velocity field.
     * @param inner_energy Function representing the specific inner energy field.
     */
    ConservativeVariablesFunction(
      const number                                         initial_time,
      const std::shared_ptr<dealii::Function<dim, number>> density,
      const std::shared_ptr<dealii::Function<dim, number>> velocity,
      const std::shared_ptr<dealii::Function<dim, number>> inner_energy);

    /**
     * Constructor for the multi-component case, where the individual primitive variable functions
     * are passed as strings and initialized as FunctionParser objects within the constructor.
     *
     * @param initial_time Initial time used to initialize the function.
     * @param density_function_string String representing the function for the density field.
     * @param velocity_function_string String representing the function for the velocity field.
     * @param inner_energy_function_string String representing the function for the specific inner energy field.
     * @param mass_fraction_functions_string String representing the functions for the mass fractions.
     * The mass fractions should be provided as a semicolon-separated list of functions.
     * @param n_mass_fractions The total number of mass fractions in the simulation.
     */
    ConservativeVariablesFunction(const number       initial_time,
                                  const std::string &density_function_string,
                                  const std::string &velocity_function_string,
                                  const std::string &inner_energy_function_string,
                                  const std::string &mass_fraction_functions_string,
                                  const unsigned     n_mass_fractions);


    /**
     * Constructor for single-species flow cases. As above the individual primitive variable
     * functions are passed as strings and initialized as FunctionParser objects within the
     * constructor.
     *
     * @param initial_time Initial time used to initialize the function.
     * @param density_function_string String representing the function for the density field.
     * @param velocity_function_string String representing the function for the velocity field.
     * @param inner_energy_function_string String representing the function for the specific inner energy field.
     */
    ConservativeVariablesFunction(const number       initial_time,
                                  const std::string &density_function_string,
                                  const std::string &velocity_function_string,
                                  const std::string &inner_energy_function_string);

    /**
     * Sets the velocity ramp-up parameters. If this function is never called or if a ramp-up
     * duration of zero is specified, no ramp-up is applied and the returned values correspond
     * directly to the fully developed velocity field.
     *
     * @param duration The duration of the ramp-up.
     * @param type The type of ramp-up function.
     */
    void
    set_velocity_ramp_up(const number duration, const RampUpType type)
    {
      velocity_ramp_up.duration = duration;
      velocity_ramp_up.type     = type;
    }

    /**
     * Set the time for the function.
     *
     * @param new_time Time to be set.
     */
    void
    set_time(const number new_time) override;

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
    value(const dealii::Point<dim, number> &loc, const unsigned int component) const override;

  private:
    /// Function describing the density field.
    const std::shared_ptr<dealii::Function<dim, number>> density;

    /// Vectorial function describing the velocity field.
    const std::shared_ptr<dealii::Function<dim, number>> velocity;

    /// Function describing the field of the specific total energy.
    const std::shared_ptr<dealii::Function<dim, number>> inner_energy;

    /// Function describing the mass fractions for multi-component flow cases.
    const std::shared_ptr<dealii::Function<dim, number>> mass_fractions;

    struct RampUpParameters
    {
      /// Ramp-up duration
      number duration{0.};

      /// Ramp-up function type
      RampUpType type{RampUpType::none};
    };

    /// Ramp-up parameters for the velocity function
    RampUpParameters velocity_ramp_up;
  };
} // namespace MeltPoolDG::CompressibleFlow