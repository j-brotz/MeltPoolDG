#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <meltpooldg/core/exceptions.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>

#include <cmath>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG
{
  /**
   * Compute the volume of a spherical particle based on its radius and the spatial dimension.
   *
   * @param radius The radius of the spherical particle.
   *
   * @throws If the function is called with dimension other than 2 or 3, an exception is thrown
   * indicating an invalid dimension.
   */
  template <int dim, typename number>
  inline number
  compute_spherical_particle_volume(const number radius)
  {
    if constexpr (dim == 3)
      {
        return 4.0 / 3.0 * M_PI * dealii::Utilities::fixed_power<3, number>(radius);
      }
    else if constexpr (dim == 2)
      {
        return M_PI * dealii::Utilities::fixed_power<2, number>(radius);
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage("Invalid dimension!"));
      }
  }

  /**
   * Compute the moment of inertia of a spherical particle based on its mass, radius, and the
   * spatial dimension.
   *
   * @param mass The mass of the spherical particle.
   * @param radius The radius of the spherical particle.
   *
   * @throws If the function is called with dimension other than 2 or 3, an exception is thrown
   * indicating an invalid dimension.
   */
  template <int dim, typename number>
  inline number
  compute_spherical_particle_moment_of_inertia(const number mass, const number radius)
  {
    if constexpr (dim == 3)
      {
        return 0.4 * mass * dealii::Utilities::fixed_power<2, number>(radius);
      }
    else if constexpr (dim == 2)
      {
        return 0.5 * mass * dealii::Utilities::fixed_power<2, number>(radius);
      }
    else
      {
        AssertThrow(false, dealii::ExcMessage("Invalid dimension!"));
      }
  }

  /**
   * Compute and set the properties of a spherical particle based on its radius and density. The
   * function computes the volume, mass, and moment of inertia of the particle using the provided
   * radius and density and sets these properties in the particle accessor.
   *
   * @param accessor The particle accessor for the spherical particle whose properties are to be
   * computed and set.
   * @param radius The radius of the spherical particle.
   * @param density The density of the spherical particle.
   */
  template <int dim, typename number>
  void
  compute_and_set_spherical_particle_properties(DEMParticleAccessor<dim, number> accessor,
                                                const number                     radius,
                                                const number                     density)
  {
    accessor.get_property(SphericalParticle<dim, number>::radius)  = radius;
    accessor.get_property(SphericalParticle<dim, number>::density) = density;
    accessor.get_property(SphericalParticle<dim, number>::volume) =
      compute_spherical_particle_volume<dim, number>(radius);
    accessor.get_property(SphericalParticle<dim, number>::mass) =
      accessor.get_property(SphericalParticle<dim, number>::density) *
      accessor.get_property(SphericalParticle<dim, number>::volume);
    accessor.get_property(SphericalParticle<dim, number>::moment_of_inertia) =
      compute_spherical_particle_moment_of_inertia<dim, number>(
        accessor.get_property(SphericalParticle<dim, number>::mass),
        accessor.get_property(SphericalParticle<dim, number>::radius));
  }

  template <int dim, typename number, typename VectorizedArrayType>
  dealii::Tensor<1, dim, VectorizedArrayType>
  vector_to_center_of_gravity(const DEMParticleAccessor<dim, number>        &particle,
                              const dealii::Point<dim, VectorizedArrayType> &location)
  {
    dealii::Point<dim, VectorizedArrayType> particle_loc;

    for (int i = 0; i < dim; ++i)
      particle_loc[i] = VectorizedArrayType(particle.get_location()[i]);

    return particle_loc - location;
  }

  template <int dim, typename number, typename VectorizedArrayType>
  dealii::Tensor<1, dim, VectorizedArrayType>
  local_particle_velocity(const DEMParticleAccessor<dim, number>        &particle,
                          const dealii::Point<dim, VectorizedArrayType> &location)
  {
    Assert(dim == 2 or dim == 3, dealii::ExcMessage("Invalid dimension!"));

    dealii::Point<dim, VectorizedArrayType> vectorized_particle_location;
    for (auto i = 0; i < dim; ++i)
      vectorized_particle_location[i] = VectorizedArrayType(particle.get_location()[i]);

    dealii::Tensor<1, dim, VectorizedArrayType> distance_to_center =
      location - vectorized_particle_location;

    dealii::Tensor<1, dim, VectorizedArrayType> velocity;
    for (unsigned int i = 0; i < dim; ++i)
      velocity[i] = VectorizedArrayType(particle.linear_velocity(i));

    dealii::Tensor<1, dim, VectorizedArrayType> angular_velocity;
    for (unsigned int i = 0; i < axial_dim<dim>; ++i)
      angular_velocity[i] = VectorizedArrayType(particle.angular_velocity(i));

    if constexpr (dim == 2)
      {
        return dealii::cross_product_2d(angular_velocity[0] * distance_to_center) + velocity;
      }
    if constexpr (dim == 3)
      {
        return velocity + dealii::cross_product_3d(angular_velocity, distance_to_center);
      }
    AssertThrow(false, dealii::ExcInternalError());
  }

  /**
   * Reads particle initial conditions (position, density, radius) from a CSV file and computes the
   * derived properties (volume, mass, moment of inertia) of each resulting spherical particle.
   *
   * The file must contain one header line, followed by one line per particle with @p dim + 2
   * comma-separated values: the particle's position, density, and radius, in that order. Blank
   * lines are skipped; any line with the wrong number of fields, or a field that cannot be parsed
   * as a number, aborts execution with a message identifying the offending line.
   *
   * @param filename Path to the particle data file. This must be an absolute path or a path
   * relative to the current working directory of the process.
   * @param mpi_communicator MPI communicator of the simulation.
   * @return The parsed particle locations and their property vectors.
   *
   * @note Only rank 0 reads and parses the file; other ranks receive empty vectors.
   */
  template <int dim, typename number>
  std::pair<std::vector<dealii::Point<dim, number>>, std::vector<std::vector<number>>>
  read_particle_state_input_file(const std::string &filename, const MPI_Comm &mpi_communicator)
  {
    enum PropertyCSVColumns
    {
      position = 0,
      density  = dim,
      radius
    };

    // Lambda function to read and parse a single line of the input file. The function reads the
    // line, computes all particle properties based on the parsed values, and returns the particle
    // properties and location as a pair. If the line is empty or contains only whitespace, it
    // returns an empty optional to indicate that no particle should be initialized from this line.
    // The function throws an exception if the line does not contain the expected number of columns
    // or if any of the values cannot be converted to a number. Note that no deal.II assertions are
    // used in this function but only standard exceptions allowing the caller to easily catch and
    // handle errors in the input file without aborting the entire program.
    const auto read_values_from_line = [](const std::string &line)
      -> std::optional<std::pair<std::vector<number>, dealii::Point<dim, number>>> {
      constexpr unsigned int n_expected_columns = dim + 2; // position (dim) + density + radius

      std::vector<std::string> parsed_strings;
      parsed_strings.reserve(n_expected_columns);

      std::istringstream data_string(line);

      std::string token;
      while (std::getline(data_string, token, ','))
        {
          parsed_strings.emplace_back(std::move(token));
        }

      // If the line does not contain any data, return an empty optional to indicate that no
      // particle should be initialized from this line. This allows for empty lines in the input
      // file without causing errors.
      if (parsed_strings.size() == 0 or
          (parsed_strings.size() == 1 and (parsed_strings[0] == "\r" or parsed_strings[0] == "\n" or
                                           parsed_strings[0] == "\t" or parsed_strings[0].empty())))
        {
          return std::nullopt;
        }

      // Check if the expected number of columns is present
      AssertThrow(parsed_strings.size() == n_expected_columns,
                  ExcInvalidCSVInputColumns(n_expected_columns, parsed_strings.size()));

      // Convert parsed strings to numbers
      std::vector<number> parsed_values;
      parsed_values.reserve(n_expected_columns);
      for (std::string &str : parsed_strings)
        {
          // remove trailing whitespace
          if (!str.empty())
            {
              char last_char = str.back();

              if (last_char == '\n' or last_char == '\r' or last_char == '\t')
                {
                  str.pop_back();
                }
            }

          std::size_t characters_processed = 0;
          parsed_values.emplace_back(std::stod(str, &characters_processed));
          AssertThrow(characters_processed == str.size(), ExcFailedToConvertStringToNumber(str));
        }

      // particle location
      dealii::Point<dim, number> particle_location;
      for (unsigned int i = 0; i < dim; ++i)
        particle_location[i] = parsed_values[PropertyCSVColumns::position + i];

      // particle properties
      std::vector<number> particle_properties(SphericalParticle<dim, number>::n_obstacle_properties,
                                              0.0);

      DEMParticleAccessor<dim, number> accessor(
        particle_location,
        particle_properties,
        dealii::numbers::invalid_unsigned_int /* dummy particle id */);

      compute_and_set_spherical_particle_properties(accessor,
                                                    parsed_values[PropertyCSVColumns::radius],
                                                    parsed_values[PropertyCSVColumns::density]);

      return std::optional<std::pair<std::vector<number>, dealii::Point<dim, number>>>(
        {particle_properties, particle_location});
    };

    std::vector<dealii::Point<dim, number>> particle_locations;
    std::vector<std::vector<number>>        particle_properties;

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::fstream file(filename);
        AssertThrow(!(file.fail()),
                    dealii::ExcMessage("Unable to open particle data file \"" + filename +
                                       "\". Aborting!"));
        std::string line;
        std::getline(file, line);

        const auto throw_error_with_line_info = [&](const unsigned int line_number,
                                                    const std::string &error_message) {
          AssertThrow(false,
                      dealii::ExcMessage("Error parsing line " + std::to_string(line_number) +
                                         " in particle data file: " + error_message));
        };

        unsigned int line_number = 2; // Start from 2 to account for the header line
        while (std::getline(file, line))
          {
            try
              {
                const std::optional<std::pair<std::vector<number>, dealii::Point<dim, number>>>
                  obstacle_properties = read_values_from_line(line);
                if (obstacle_properties.has_value())
                  {
                    particle_properties.push_back(obstacle_properties.value().first);
                    particle_locations.push_back(obstacle_properties.value().second);
                  }
                ++line_number;
              }
            catch (const ExcInvalidCSVInputColumns &e)
              {
                throw_error_with_line_info(line_number, e.what());
              }
            catch (const ExcFailedToConvertStringToNumber &e)
              {
                throw_error_with_line_info(line_number, e.what());
              }
          }
      }

    return {std::move(particle_locations), std::move(particle_properties)};
  }
} // namespace MeltPoolDG
