#include <deal.II/base/exceptions.h>

#include <meltpooldg/melt_pool/powder_bed.hpp>

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <sstream>

namespace MeltPoolDG::MeltPool
{
  using namespace dealii;

  void
  PowderBedData::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("powder bed");
    {
      prm.add_parameter("particle list file",
                        particle_list_file,
                        "Specify file containing the list of particles.");
      prm.add_parameter("substrate level",
                        substrate_level,
                        "This parameter controls the level of the substrate in the vertical axis.");
      prm.add_parameter(
        "slice location",
        slice_location,
        "In a 2D simulation, this parameter controls the location of the slice plane through "
        "the powder bed along the y-axis in particle space.");
    }
    prm.leave_subsection();
  }

  template <int dim>
  PowderBedLevelSet<dim>::PowderBedLevelSet(const PowderBedData &powder_bed_data,
                                            const LevelSetType   level_set_type,
                                            const double         eps)
    : Function<dim>()
    , data(powder_bed_data)
    , particles(read_particles_from_file(data.particle_list_file))
    , level_set_type(level_set_type)
    , eps(eps)
  {
    AssertThrow(!particles.empty(),
                ExcMessage("The powder bed contains no powder particles! "
                           "Please check input files."));
  }

  template <int dim>
  double
  PowderBedLevelSet<dim>::value(const Point<dim> &p, const unsigned int /* component */) const
  {
    // compute the signed distance in particle space, which is always 3D

    Point<3> point;
    if constexpr (dim == 3)
      point = p;
    else if (dim == 2)
      {
        // convert 2D space into particle space:
        // both x-axis coincide
        point[0] = p[0];
        // slice plane location specified by input parameter
        point[1] = data.slice_location;
        // y-axis of 2D plane coincides with z-axis in particle space
        point[2] = p[1];
      }
    else
      AssertThrow(false, ExcNotImplemented());

    double signed_distance = std::numeric_limits<double>::lowest();
    for (const auto &particle : particles)
      signed_distance = std::max(signed_distance,
                                 /* signed distance to particle */ particle.second -
                                   point.distance(particle.first));

    if (data.substrate_level != std::numeric_limits<double>::lowest())
      signed_distance =
        std::max(signed_distance,
                 /* signed distance to substrate */ data.substrate_level - point[2]);

    switch (level_set_type)
      {
        case LevelSetType::level_set:
          return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
            signed_distance, eps);
        case LevelSetType::heaviside:
          return UtilityFunctions::CharacteristicFunctions::heaviside(signed_distance, eps);
        case LevelSetType::signed_distance:
          return signed_distance;
        default:
          AssertThrow(false, ExcNotImplemented());
      }
    // unreachable dummy return
    return 0.0;
  }

  template <int dim>
  std::vector<typename PowderBedLevelSet<dim>::Particle>
  PowderBedLevelSet<dim>::read_particles_from_file(const std::string &input_file) const
  {
    auto particle_list_file_stream = open_file(input_file);

    // csv particle output format
    auto temp = read_particles_from_csv_output(particle_list_file_stream);

    // TODO prune particles that are out of bounds
    return temp;
  }

  template <int dim>
  std::ifstream
  PowderBedLevelSet<dim>::open_file(const std::string &input_file) const
  {
    std::filesystem::path input_file_path(input_file);

    if (not std::filesystem::exists(input_file_path))
      {
        // if file not found, try relative to source directory (for testing)
        const auto relative_to_source = std::filesystem::path(SOURCE_DIR) / input_file_path;
        if (std::filesystem::exists(relative_to_source))
          input_file_path = relative_to_source;
        else
          AssertThrow(false,
                      ExcMessage("Input file <" + input_file +
                                 "> not found. Please make sure the file exists!"));
      }

    std::ifstream input_file_stream(input_file_path);
    if (!input_file_stream)
      {
        input_file_stream.close();
        AssertThrow(false,
                    ExcMessage("Input file <" + input_file +
                               "> not found. Please make sure the file exists!"));
      }
    return input_file_stream;
  }

  template <int dim>
  std::vector<typename PowderBedLevelSet<dim>::Particle>
  PowderBedLevelSet<dim>::read_particles_from_csv_output(std::ifstream &csv_file_stream) const
  {
    std::vector<Particle> particle_vector;

    // count number of lines in csv
    const auto n_lines = std::count(std::istreambuf_iterator<char>(csv_file_stream),
                                    std::istreambuf_iterator<char>(),
                                    '\n');
    AssertThrow(n_lines > 0, ExcMessage("The particle list file has no entries!"));
    // ... go back to beginning of the file stream
    csv_file_stream.clear();
    csv_file_stream.seekg(0);
    // ... reserve space in particle vector
    particle_vector.reserve(n_lines - 1);

    // convert to meter
    const double unit_conversion_factor = 1e-3;

    // buffers
    Point<3>    coordinates;
    double      radius;
    std::string line, cell;

    // skip header
    std::getline(csv_file_stream, line);

    while (std::getline(csv_file_stream, line))
      {
        AssertThrow(std::count(line.begin(), line.end(), ',') == 5,
                    ExcMessage("Invalid csv file, the number of columns must be 6!"));

        try
          {
            std::stringstream line_stream(line);
            // skip first two columns
            std::getline(line_stream, cell, ',');
            std::getline(line_stream, cell, ',');
            // read coordinates
            for (int i = 0; i < 3; ++i)
              {
                std::getline(line_stream, cell, ',');
                coordinates[i] = std::stod(cell) * unit_conversion_factor;
              }
            // read radius
            std::getline(line_stream, cell, ',');
            radius = std::stod(cell) * unit_conversion_factor;
          }
        catch (...)
          {
            AssertThrow(
              false,
              ExcMessage(
                "Invalid csv file! Besides a header line the csv file must only consist of "
                "numbers in 6 columns separated by commas."));
          }

        particle_vector.emplace_back(coordinates, radius);
      }

    return particle_vector;
  }

  template class PowderBedLevelSet<1>;
  template class PowderBedLevelSet<2>;
  template class PowderBedLevelSet<3>;
} // namespace MeltPoolDG::MeltPool
