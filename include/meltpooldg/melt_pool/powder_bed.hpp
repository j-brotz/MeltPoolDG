#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <meltpooldg/utilities/better_enum.hpp>
#include <meltpooldg/utilities/utility_functions.hpp>

#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace MeltPoolDG::MeltPool
{
  template <typename number>
  struct PowderBedData
  {
    std::string particle_list_file;
    number      substrate_level = std::numeric_limits<number>::lowest();
    number      slice_location  = 0.0;

    void
    add_parameters(dealii::ParameterHandler &prm);
  };

  BETTER_ENUM(LevelSetType, char, level_set, heaviside, signed_distance)

  template <int dim, typename number>
  class PowderBedLevelSet : public dealii::Function<dim, number>
  {
    // pair of center and radius
    using Particle = std::pair<dealii::Point<3>, number>;

  public:
    PowderBedLevelSet(const PowderBedData<number> &powder_bed_data,
                      const LevelSetType           level_set_type = LevelSetType::level_set,
                      const number                 eps            = 0.0);

    number
    value(const dealii::Point<dim> &p, const unsigned int /* component */) const override;

  private:
    std::vector<Particle>
    read_particles_from_file(const std::string &input_file) const;

    std::ifstream
    open_file(const std::string &input_file) const;

    /**
     * particle list format:
     *
     * format:      csv
     * header:      yes
     * columns:     6
     * coordinates: columns 3-5
     * radius:      column 6
     * unit:        mm
     * orientation: z-axis points up
     */
    std::vector<Particle>
    read_particles_from_csv_output(std::ifstream &csv_file_stream) const;

    const PowderBedData<number> data;
    const std::vector<Particle> particles;
    const LevelSetType          level_set_type;
    const number                eps;
  };
} // namespace MeltPoolDG::MeltPool